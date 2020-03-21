#include "ctc_beam_search_decoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <utility>
// for debugging
#include <iostream>
#include <chrono>

#include "ThreadPool.h"
#include "fst/fstlib.h"

#include "decoder_utils.h"
#include "path_trie.h"

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;

std::vector<std::pair<double, std::string>> ctc_beam_search_decoder(
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer) {
  // dimension check
  size_t num_time_steps = probs_seq.size();
  for (size_t i = 0; i < num_time_steps; ++i) {
    VALID_CHECK_EQ(probs_seq[i].size(),
                   vocabulary.size() + 1,
                   "The shape of probs_seq does not match with "
                   "the shape of the vocabulary");
  }

  // assign blank id
  // DCMMC: blank id is 0 for chineseocr
  // size_t blank_id = vocabulary.size();
  size_t blank_id = 0;

  // assign space id
  auto it = std::find(vocabulary.begin(), vocabulary.end(), " ");
  // DCMMC: ids of all characters must plus 1 for chineseocr
  // int space_id = it - vocabulary.begin();
  int space_id = it - vocabulary.begin() + 1;
  // if no space in vocabulary
  // if ((size_t)space_id >= vocabulary.size()) {
  if ((size_t)space_id > vocabulary.size()) {
    space_id = -2;
  }
  // std::cout << "space id: " << space_id << std::endl;

  // init prefixes' root
  PathTrie root;
  root.score = root.log_prob_b_prev = 0.0;
  std::vector<PathTrie *> prefixes;
  prefixes.push_back(&root);

  // std::cout << "prefixed initialized!" << std::endl;

  if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
    auto fst_dict = static_cast<fst::StdVectorFst *>(ext_scorer->dictionary);
    fst::StdVectorFst *dict_ptr = fst_dict->Copy(true);
    root.set_dictionary(dict_ptr);
    auto matcher = std::make_shared<FSTMATCH>(*dict_ptr, fst::MATCH_INPUT);
    root.set_matcher(matcher);
  }

  // std::cout << "Start prefix search over time" << std::endl;

  // prefix search over time
  auto start_voca = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::high_resolution_clock::now() - start_voca;
  auto seconds = 0.001 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  for (size_t time_step = 0; time_step < num_time_steps; ++time_step) {
    auto &prob = probs_seq[time_step];
    // std::cout << "time step " << time_step << std::endl;

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;
    if (ext_scorer != nullptr) {
      size_t num_prefixes = std::min(prefixes.size(), beam_size);
      std::sort(
          prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);
      min_cutoff = prefixes[num_prefixes - 1]->score +
                   std::log(prob[blank_id]) - std::max(0.0, ext_scorer->beta);
      full_beam = (num_prefixes == beam_size);
    }

    std::vector<std::pair<size_t, float>> log_prob_idx =
        get_pruned_log_probs(prob, cutoff_prob, cutoff_top_n);

    // std::cout << "top 3 prob: (" << log_prob_idx[0].first << ", " << log_prob_idx[0].second
    //     << "), (" << log_prob_idx[1].first << ", " << log_prob_idx[1].second << "), ("
    //     << log_prob_idx[2].first << ", " << log_prob_idx[2].second << ")." << std::endl;

    // std::cout << "Start loop over chars" << std::endl;
    // loop over chars
    for (size_t index = 0; index < log_prob_idx.size(); index++) {
      auto c = log_prob_idx[index].first;
      auto log_prob_c = log_prob_idx[index].second;

      for (size_t i = 0; i < prefixes.size() && i < beam_size; ++i) {
        auto prefix = prefixes[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
          break;
        }
        // blank
        if (c == blank_id) {
          prefix->log_prob_b_cur =
              log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
          continue;
        }
        // repeated character
        if (c == prefix->character) {
          prefix->log_prob_nb_cur = log_sum_exp(
              prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
        }
        // get new prefix
        auto prefix_new = prefix->get_path_trie(c);

        // std::cout << "prefix_new" << std::endl;

        if (prefix_new != nullptr) {
          float log_p = -NUM_FLT_INF;

          if (c == prefix->character &&
              prefix->log_prob_b_prev > -NUM_FLT_INF) {
            log_p = log_prob_c + prefix->log_prob_b_prev;
          } else if (c != prefix->character) {
            log_p = log_prob_c + prefix->score;
          }

          // std::cout << "before LM" << std::endl;
          // language model scoring
          if (ext_scorer != nullptr &&
              (c == space_id || ext_scorer->is_character_based())) {
            PathTrie *prefix_to_score = nullptr;
            // skip scoring the space
            if (ext_scorer->is_character_based()) {
              prefix_to_score = prefix_new;
            } else {
              prefix_to_score = prefix;
            }

            float score = 0.0;
            std::vector<std::string> ngram;
            ngram = ext_scorer->make_ngram(prefix_to_score);
            // std::cout << "get ngram done" << std::endl;
            score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
            // std::cout <<"get log cond prob done" << std::endl;
            log_p += score;
            log_p += ext_scorer->beta;
          }
          // std::cout << "after LM" << std::endl;
          prefix_new->log_prob_nb_cur =
              log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
      }  // end of loop over prefix
    }    // end of loop over vocabulary

    prefixes.clear();
    // update log probs
    root.iterate_to_vec(prefixes);

    // only preserve top beam_size prefixes
    if (prefixes.size() >= beam_size) {
      std::nth_element(prefixes.begin(),
                       prefixes.begin() + beam_size,
                       prefixes.end(),
                       prefix_compare);
      for (size_t i = beam_size; i < prefixes.size(); ++i) {
        prefixes[i]->remove();
      }
    }

    auto max_prefix = *std::max_element(prefixes.begin(), prefixes.end(),
        [](const PathTrie * s1, const PathTrie * s2) {
        return s1->score < s2->score;
            });
    // std::cout << "best beam: " << max_prefix->character << " ("
    //     << vocabulary[max_prefix->character - 1]
    //     << ")." << std::endl;
    // elapsed = std::chrono::high_resolution_clock::now() - start_voca;
    // seconds = 0.000001 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    // std::cout << "Time to take " << time_step << " time_steps for " << beam_size << " beams: " << seconds << "s." << std::endl;
  }  // end of loop over time
  auto elapsed_voca = std::chrono::high_resolution_clock::now() - start_voca;
  auto seconds_voca = 0.000001 * std::chrono::duration_cast<std::chrono::microseconds>(elapsed_voca).count();
  std::cout << "Time to take all vocabulary: " << seconds_voca << "s." << std::endl;

  // score the last word of each prefix that doesn't end with space
  if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
    for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
      auto prefix = prefixes[i];
      if (!prefix->is_empty() && prefix->character != space_id) {
        float score = 0.0;
        std::vector<std::string> ngram = ext_scorer->make_ngram(prefix);
        score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
        score += ext_scorer->beta;
        prefix->score += score;
      }
    }
  }

  size_t num_prefixes = std::min(prefixes.size(), beam_size);
  std::sort(prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);

  // compute aproximate ctc score as the return score, without affecting the
  // return order of decoding result. To delete when decoder gets stable.
  for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
    double approx_ctc = prefixes[i]->score;
    if (ext_scorer != nullptr) {
      std::vector<int> output;
      prefixes[i]->get_path_vec(output);
      auto prefix_length = output.size();
      auto words = ext_scorer->split_labels(output);
      // remove word insert
      approx_ctc = approx_ctc - prefix_length * ext_scorer->beta;
      // remove language model weight:
      approx_ctc -= (ext_scorer->get_sent_log_prob(words)) * ext_scorer->alpha;
    }
    prefixes[i]->approx_ctc = approx_ctc;
  }

  // seconds_voca = 0.000001 * std::chrono::duration_cast<std::chrono::microseconds>(
  //     std::chrono::high_resolution_clock::now() - start_voca - elapsed_voca).count();
  // std::cout << "Time to postprocess: " << seconds_voca << std::endl;
  return get_beam_search_result(prefixes, vocabulary, beam_size);
}


std::vector<std::vector<std::pair<double, std::string>>>
ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer) {
  // std::cout << "start debugging" << std::endl;
  VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
  // thread pool
  ThreadPool pool(num_processes);
  // number of samples
  size_t batch_size = probs_split.size();

  // std::cout << "Start enqueue th tasks into thread pool" << std::endl;
  // enqueue the tasks of decoding
  std::vector<std::future<std::vector<std::pair<double, std::string>>>> res;
  for (size_t i = 0; i < batch_size; ++i) {
    res.emplace_back(pool.enqueue(ctc_beam_search_decoder,
                                  probs_split[i],
                                  vocabulary,
                                  beam_size,
                                  cutoff_prob,
                                  cutoff_top_n,
                                  ext_scorer));
  }

  // get decoding results
  std::vector<std::vector<std::pair<double, std::string>>> batch_results;
  for (size_t i = 0; i < batch_size; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}

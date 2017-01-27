
#include <vector>
#include <string>
#include <unordered_map>

// NOTE: features is the return vector.  we assume it is preinitialized
// to length(nblocks) with 0.0

const std::string DIV = "div";
const std::string P = "p";

void _readability_features(
    std::vector<uint32_t>& block_text_len,
    std::vector<std::vector<std::pair<uint32_t, int> > >& block_readability_class_weights,
    std::vector<std::vector<uint32_t> >& block_ancestors,
    std::vector<std::string>& block_start_tag,
    std::vector<double>& block_link_density,
    int& nblocks,
    double* features)
{
    std::unordered_map<uint32_t, double> scores;
    // link density in subtrees.
    // first entry is total text length * link density,
    // second is total text length
    std::unordered_map<uint32_t, std::pair<double, double> > ld;
    // only tag_ids that have <p> or <div> as children are valid root nodes
    std::unordered_map<uint32_t, bool> valid_nodes;

    std::vector<std::pair<uint32_t, int> >::const_iterator it_cw;
    std::vector<uint32_t>::const_iterator it_ancestors;
    std::unordered_map<uint32_t, double>::iterator it_scores;
    for (std::size_t k = 0; k < nblocks; ++k)
    {
        //  1. create content_score for each tag_id
        //  read through blocks.  for each class weight written, add its weight
        //       then: if text length > 25:
        //        add in min((text_len / 100), 3) * link density to
        //            to the parent
        for (it_cw = block_readability_class_weights[k].begin();
            it_cw != block_readability_class_weights[k].end(); ++it_cw)
            scores[it_cw->first] = it_cw->second;

        if (block_ancestors[k].size() > 0)
        {
            for (it_ancestors = block_ancestors[k].begin();
                it_ancestors != block_ancestors[k].end(); ++it_ancestors)
            {
                uint32_t ancestor = *it_ancestors;
                if (ld.find(ancestor) == ld.end())
                    ld[ancestor] = std::make_pair(0.0, 0.0);
                ld[ancestor].first += block_link_density[k] * block_text_len[k];
                ld[ancestor].second += block_text_len[k];
            }

            if (block_text_len[k] > 25 && 
                (block_start_tag[k] == DIV || block_start_tag[k] == P))
            {
                uint32_t parent = block_ancestors[k].back();
                scores[parent] += (
                    1 + std::min(int(block_text_len[k] / 100), 3));
                valid_nodes[parent] = true;
            }
        }
    }

    // scale scores by link density
    for (it_scores = scores.begin(); it_scores != scores.end();
        ++it_scores)
    {
        uint32_t tag_id = it_scores->first;
        it_scores->second *= (
            1.0 - ld[tag_id].first / std::max(ld[tag_id].second, 1.0));
    }

    // 2. get max score of all valid scores
    bool a_valid_score = false;
    double max_score = -1.0e20;
    for (it_scores = scores.begin(); it_scores != scores.end();
        ++it_scores)
    {
        uint32_t tag_id = it_scores->first;
        if (valid_nodes.find(tag_id) != valid_nodes.end())
        {
            a_valid_score = true;
            max_score = std::max(max_score, it_scores->second);
        }
    }
    max_score = std::max(max_score, 1.0);

    // if we don't have a valid score, then all feature are 0.0.
    if (!a_valid_score)
        return;

    // 3. read through blocks again.  for each ancestor,
    //  get max scores of all ancestors and store feature as max
    //  ancestor score / max score
    for (std::size_t k = 0; k < nblocks; ++k)
    {
        if (block_ancestors[k].empty())
            features[k] = 0.0;
        else
        {
            // check for valid ancestors and scores.  get the max
            // among ancestors
            double block_max = -1e20;
            bool a_valid_ancestor = false;
            for (it_ancestors = block_ancestors[k].begin();
                it_ancestors != block_ancestors[k].end(); ++it_ancestors)
            {
                uint32_t tag_id = *it_ancestors;
                if (valid_nodes.find(tag_id) != valid_nodes.end())
                {
                    a_valid_ancestor = true;
                    block_max = std::max(block_max, scores[tag_id]);
                }
                if (a_valid_ancestor)
                    features[k] = std::max(block_max / max_score, 0.0);
                else
                    features[k] = 0.0;
            }
        }
    }
}


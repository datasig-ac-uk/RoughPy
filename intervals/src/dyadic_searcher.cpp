// Copyright (c) 2023 RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 03/03/23.
//

#include "dyadic_searcher.h"

#include <algorithm>

using namespace rpy::intervals;

void DyadicSearcher::expand_left(ScaledPredicate& predicate,
                                 std::deque<DyadicInterval>& current) const
{
    auto di = current.front();
    bool is_aligned = di.aligned();
    --di;// moving to the left

    if (is_aligned) {
        // It's possible that the interval contains two dyadics of starting
        // length if and only if the left hand is not aligned and the right
        // hand is aligned.
        if (predicate(di)) {
            current.push_front(di);
            --di;
        }
    }
    /*
     * At this stage, we should be in the situation where predicate(di) is
     * false. So we should split the interval in half, and check the right hand
     * interval.
     */
    while (di.power() < m_max_depth) {
        DyadicInterval left(di);
        left.shrink_interval_left();
        di.shrink_interval_right();

        // Check new di
        if (predicate(di)) {
            current.push_front(di);

            // If the right hand child is good then move to the left
            // and repeat. Note left cannot be good if right is good.
            di = left;
        }

        /*
         * If di was not good here then we continue with di and discard
         * left, since it cannot be part of the same interval.
         */
    }
}
void DyadicSearcher::expand_right(ScaledPredicate& predicate,
                                  std::deque<DyadicInterval>& current) const
{
    auto di = current.back();
    /*
     * We don't need to check if our neighbour at the same level is good here
     * because in the main search we're moving right to left, so we would have
     * found that interval first and expanded into this one during expand_left.
     */
    ++di;// moving to the right.

    while (di.power() < m_max_depth) {
        DyadicInterval right(di);
        right.shrink_interval_right();
        di.shrink_interval_left();

        // Check current di
        if (predicate(di)) {
            current.push_back(di);

            /*
             * If the current di is included, then we shuffle to the right
             * and start looking there.
             */
            di = right;
        }
        /*
         * If di is not good then we continue searching here and discard
         * right, since that cannot be part of the same interval.
         */
    }
}
void DyadicSearcher::expand(ScaledPredicate& predicate,
                            DyadicInterval found_interval)
{
    std::deque<DyadicInterval> found{std::move(found_interval)};
    expand_left(predicate, found);
    expand_right(predicate, found);

    m_seen[found.back().dsup()] = found.front().dinf();
}
ScaledPredicate
DyadicSearcher::rescale_to_unit_interval(const Interval& original)
{
    auto a = original.inf();
    auto b = original.sup();

    RPY_CHECK(b > a);// a degenerate interval is going to be a pain

    /*
     * The transformation of original onto the unit interval is given by
     *
     * x -> (x - a)/(b - a).
     *
     * The scaled_predicate needs to know about the inverse map which is
     * give by
     *
     * y -> a + y*(b-a)
     */

    return {m_predicate, a, (b - a)};
}
void DyadicSearcher::get_next_dyadic(DyadicInterval& current) const
{
    DyadicRealStrictLess diless;
    DyadicRealStrictGreater digreater;

    auto k = current.multiplier();
    const auto n = current.power();
    const auto itype = current.type();

    for (const auto& pair : m_seen) {
        Dyadic current_dyadic{k, n};
        /*
         * Seen pairs are of the form (sup, inf) rather than (inf, sup).
         * Check current.inf >= pair.inf, and if so that current.inf <=
         * pair.sup. In this case, we can skip to the first interval I with
         * I.sup < pair.inf.
         */
        if (!diless(current_dyadic, pair.second)) {
            /*
             * Current is equal or right of pair.inf, so we need to check
             * if it overlaps.
             */
            if (!digreater(current_dyadic, pair.first)) {

                if (n < pair.second.power()) {
                    /*
                     * First case, when pair.inf has higher resolution than
                     * current. This case is much more likely to be the case. In
                     * this case, first get the parent of pair.inf in the
                     * resolution of current. Then find the interval to the
                     * left.
                     */
                    k = (pair.second.multiplier() >> (pair.second.power() - n));
                } else if (n > pair.second.power()) {
                    /*
                     * Second case, when pair.inf has a lower resolution than
                     * current. This case is unlikely but still possible. In
                     * this case, rebase pair.info to the resolution of current,
                     * and then decrement.
                     */
                    k = (pair.second.multiplier() << (n - pair.second.power()));
                } else {
                    /*
                     * Third case, when pair.inf and current have the same
                     * resolution. Very unlikely, but still possible. In this
                     * case, just decrement pair.inf.
                     */
                    k = pair.second.multiplier();
                }
            } else {
                /*
                 * If it doesn't overlap then, because of the ordering in
                 * m_seen, we can stop looking here.
                 */
                break;
            }
        }
    }

    /*
     * If we got here then either next < pair.inf for all of the pairs we've
     * seen or, next >= pair.sup for all of the intervals.
     */
    --k;

    current = {k, n, itype};
}
std::vector<RealInterval>
DyadicSearcher::find_in_unit_interval(ScaledPredicate& predicate)
{
    //    assert(!predicate(dyadic_interval(0, 0)));
    m_seen.clear();

    for (dyadic_depth_t current_depth = 1; current_depth <= m_max_depth;
         ++current_depth) {
        // Starting interval for this depth is [(2^d-1)/2^d, 2^d/2^d)
        DyadicInterval check_current{(1 << current_depth), current_depth};
        get_next_dyadic(check_current);

        // We're iterating from right to left, so the end point is 0.
        while (check_current.multiplier() >= 0) {
            bool is_good = predicate(check_current);
            if (is_good) {
                /*
                 * Current interval is good, expand this as far as allows
                 * and put the beginning and end of the found interval in
                 * the seen map.
                 */
                expand(predicate, check_current);
            }

            // Calculate the next interval
            get_next_dyadic(check_current);
        }
    }
    std::vector<RealInterval> result;
    result.reserve(m_seen.size());
    for (const auto& pair : m_seen) {
        result.emplace_back(static_cast<double>(pair.second),
                            static_cast<double>(pair.first));
    }
    std::reverse(result.begin(), result.end());
    return result;
}
std::vector<RealInterval> DyadicSearcher::operator()(const Interval& original)
{
    if (m_predicate(original)) {
        // If the original interval is good, there is nothing to do.
        return {RealInterval(original)};
    }

    auto predicate = rescale_to_unit_interval(original);
    auto found = find_in_unit_interval(predicate);

    std::vector<RealInterval> result;
    result.reserve(found.size());
    for (const auto& itvl : found) {
        result.emplace_back(predicate.unscale(itvl));
    }
    return result;
}

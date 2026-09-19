#include "brain/opening_book.h"

#include <algorithm>
#include <cstring>

#if __has_include("brain/opening_book_data.h")
#include "brain/opening_book_data.h"
#define SENTRY_HAS_BOOK 1
#else
#define SENTRY_HAS_BOOK 0
#endif

namespace brain {

#if SENTRY_HAS_BOOK

// 记录 = 9 字节：pr_lo pr_hi pb_lo pb_hi turn side ac freef best
struct BookEntry {
    int pr, pb, turn, side, ac, freef, best;
};

// 从扁平数组重建结构化条目（编译期一次性，查表用二分）
static BookEntry g_entries[kBookN];
static bool g_init = false;

static void init_entries() {
    if (g_init) return;
    for (int i = 0; i < kBookN; ++i) {
        const unsigned char* r = kBookData + i * 9;
        g_entries[i].pr = r[0] | (r[1] << 8);
        g_entries[i].pb = r[2] | (r[3] << 8);
        g_entries[i].turn = r[4];
        g_entries[i].side = r[5];
        g_entries[i].ac = r[6];
        g_entries[i].freef = r[7];
        g_entries[i].best = r[8];
    }
    std::sort(g_entries, g_entries + kBookN, [](const BookEntry& a, const BookEntry& b) {
        if (a.pr != b.pr) return a.pr < b.pr;
        if (a.pb != b.pb) return a.pb < b.pb;
        if (a.turn != b.turn) return a.turn < b.turn;
        if (a.side != b.side) return a.side < b.side;
        if (a.ac != b.ac) return a.ac < b.ac;
        if (a.freef != b.freef) return a.freef < b.freef;
        return false; // 同键不同 diff 不区分（开局期分差恒 0）
    });
    g_init = true;
}

#endif // SENTRY_HAS_BOOK

bool book_available() {
#if SENTRY_HAS_BOOK
    return kBookN > 0;
#else
    return false;
#endif
}

int book_turns() {
#if SENTRY_HAS_BOOK
    return kBookTurns;
#else
    return 0;
#endif
}

int book_lookup(int pr, int pb, int turn, int side, int ac, int freef, int /*diff*/) {
#if SENTRY_HAS_BOOK
    init_entries();
    // 二分：键 = (pr, pb, turn, side, ac, freef)，diff 不参与（开局期恒 0）
    int lo = 0, hi = kBookN - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        const BookEntry& e = g_entries[mid];
        if (e.pr < pr || (e.pr == pr && e.pb < pb) ||
            (e.pr == pr && e.pb == pb && e.turn < turn) ||
            (e.pr == pr && e.pb == pb && e.turn == turn && e.side < side) ||
            (e.pr == pr && e.pb == pb && e.turn == turn && e.side == side && e.ac < ac) ||
            (e.pr == pr && e.pb == pb && e.turn == turn && e.side == side &&
             e.ac == ac && e.freef < freef)) {
            lo = mid + 1;
        } else if (e.pr == pr && e.pb == pb && e.turn == turn && e.side == side &&
                   e.ac == ac && e.freef == freef) {
            return e.best;
        } else {
            hi = mid - 1;
        }
    }
#endif
    return -1;
}

} // namespace brain

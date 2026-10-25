// Host-side test of onjuino/play_ring.h (the rolling playback buffer logic).
// Built and run by tests/test_play_ring.py:  python3 tests/test_play_ring.py
//
// The simulation at the bottom mirrors the pass structure of opusDecodeTask in
// onjuino.ino (intake -> play one 32 ms chunk -> stall rule) on the same
// PlayRing / OpusFrameRx / PlayGate primitives, with simulated time.
#include "../onjuino/play_ring.h"

#include <stdio.h>
#include <string.h>
#include <deque>
#include <vector>

static int passed = 0, failed = 0;
#define CHECK(name, cond)                                  \
    do                                                     \
    {                                                      \
        if (cond) { passed++; printf("PASS  %s\n", name); } \
        else { failed++; printf("FAIL  %s  (line %d)\n", name, __LINE__); } \
    } while (0)

static void test_ring()
{
    int16_t store[8];
    PlayRing r;
    r.begin(store, 8);
    int16_t in[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    CHECK("push limited by free space", r.push(in, 10) == 8 && r.count == 8 && r.freeSpace() == 0);
    int16_t out[10];
    CHECK("pop returns oldest first", r.pop(out, 3) == 3 && out[0] == 1 && out[2] == 3 && r.count == 5);
    CHECK("push after pop wraps", r.push(in + 8, 2) == 2 && r.count == 7);
    CHECK("pop limited by count", r.pop(out, 10) == 7);
    int16_t expect[7] = {4, 5, 6, 7, 8, 9, 10};
    CHECK("wrapped order preserved", memcmp(out, expect, sizeof(expect)) == 0);
    CHECK("empty after drain", r.count == 0 && r.pop(out, 1) == 0);
    r.push(in, 5);
    r.clear();
    CHECK("clear empties", r.count == 0 && r.freeSpace() == 8);
    // many wraps with odd sizes keep the sequence intact
    int16_t seq = 0, want = 0;
    bool ok = true;
    for (int i = 0; i < 1000; i++)
    {
        int16_t tmp[5];
        for (int k = 0; k < 5; k++) tmp[k] = seq + k;
        size_t n = r.push(tmp, 5);
        seq += n;
        int16_t o[3];
        size_t m = r.pop(o, 3);
        for (size_t k = 0; k < m; k++) ok &= (o[k] == want++);
    }
    CHECK("1000 wraps keep FIFO order", ok);
}

static void feed_bytes(OpusFrameRx &rx, const std::vector<uint8_t> &bytes, size_t step,
                       std::vector<OpusRxEvent> &events, std::vector<size_t> &lens)
{
    size_t i = 0;
    while (i < bytes.size())
    {
        size_t n = rx.wanted();
        if (n > step) n = step;
        if (n > bytes.size() - i) n = bytes.size() - i;
        memcpy(rx.dst(), &bytes[i], n);
        i += n;
        OpusRxEvent ev = rx.feed(n);
        if (ev != OPUS_RX_MORE) { events.push_back(ev); lens.push_back(rx.frameLen); }
        if (ev == OPUS_RX_END || ev == OPUS_RX_BAD_LEN) break;
    }
}

static void test_parser()
{
    std::vector<uint8_t> stream;
    for (int f = 0; f < 3; f++)
    {
        size_t len = 10 + f * 7;
        stream.push_back(len >> 8);
        stream.push_back(len & 0xFF);
        for (size_t k = 0; k < len; k++) stream.push_back((uint8_t)(f * 50 + k));
    }
    stream.push_back(0);
    stream.push_back(0); // end marker
    for (size_t step : {1u, 2u, 3u, 7u, 1000u})
    {
        uint8_t pkt[100];
        OpusFrameRx rx;
        rx.begin(pkt, sizeof(pkt));
        std::vector<OpusRxEvent> ev;
        std::vector<size_t> lens;
        feed_bytes(rx, stream, step, ev, lens);
        char name[80];
        snprintf(name, sizeof(name), "parser: 3 frames then END (read size %zu)", step);
        CHECK(name, ev.size() == 4 && ev[0] == OPUS_RX_FRAME && ev[1] == OPUS_RX_FRAME &&
                        ev[2] == OPUS_RX_FRAME && ev[3] == OPUS_RX_END && lens[0] == 10 &&
                        lens[1] == 17 && lens[2] == 24);
    }
    uint8_t pkt[100];
    OpusFrameRx rx;
    rx.begin(pkt, 100);
    std::vector<OpusRxEvent> ev;
    std::vector<size_t> lens;
    feed_bytes(rx, {0x01, 0xF4}, 1, ev, lens); // length 500 > 100
    CHECK("parser: oversize length is BAD_LEN", ev.size() == 1 && ev[0] == OPUS_RX_BAD_LEN);
    rx.begin(pkt, 100);
    CHECK("parser: idle is not mid-frame", !rx.midFrame());
    rx.dst()[0] = 0;
    rx.feed(1);
    CHECK("parser: half a length is mid-frame", rx.midFrame() && rx.wanted() == 1);
    rx.dst()[0] = 5;
    rx.feed(1);
    CHECK("parser: frame body wanted after length", rx.midFrame() && rx.wanted() == 5);
}

static void test_gate()
{
    PlayGate g;
    CHECK("gate: below threshold waits", !g.mayPlay(100, 1000, false));
    CHECK("gate: empty stream that ended plays nothing", !g.mayPlay(0, 1000, true) && !g.started);
    CHECK("gate: starts at threshold", g.mayPlay(1000, 1000, false));
    CHECK("gate: stays started below threshold", g.mayPlay(50, 1000, false));
    CHECK("gate: empty ring has nothing to play", !g.mayPlay(0, 1000, false) && g.started);
    PlayGate s;
    CHECK("gate: short stream plays at stream end", !s.mayPlay(300, 1000, false) && s.mayPlay(300, 1000, true));
}

// ---- simulation -------------------------------------------------------------
// Time advances 32 ms per pass (one i2s_write of a 512-sample chunk). The server
// sends one 320-sample "frame" (60 bytes + 2 length) per 20 ms of audio as fast
// as the pod accepts it, except during an outage [outageStart, outageEnd) when
// nothing is delivered. Samples carry a running counter to check order and loss.
struct SimResult
{
    bool aborted = false;      // stall rule fired
    size_t played = 0;
    bool inOrder = true;
    int underruns = 0;        // started, not done, ring empty at the play step
    size_t minCushionAfter = (size_t)-1; // smallest ring.count seen after 'warm' ms of playback
};

static SimResult simulate(size_t thresholdSamples, size_t totalFrames, uint32_t outageStartMs,
                          uint32_t outageEndMs, uint32_t warmMs)
{
    const size_t FRAME = 320, CHUNK = 512, HEADROOM = FRAME + 256;
    std::vector<int16_t> store(thresholdSamples + HEADROOM);
    PlayRing ring;
    ring.begin(store.data(), store.size());
    uint8_t pkt[4000];
    OpusFrameRx rx;
    rx.begin(pkt, sizeof(pkt));
    PlayGate gate;

    std::deque<uint8_t> tcp; // bytes the server has delivered to the pod
    size_t framesSent = 0;
    int16_t nextSample = 0;
    int16_t wantSample = 0;
    bool streamDone = false;
    uint32_t now = 0, lastProgress = 0, playStart = 0;
    const uint32_t STALL_MS = 5000;
    SimResult res;

    for (int pass = 0; pass < 200000; pass++, now += 32)
    {
        bool outage = now >= outageStartMs && now < outageEndMs;
        // server side: keep the pod's receive window (~4 KB) full unless in outage
        while (!outage && framesSent <= totalFrames && tcp.size() < 4096)
        {
            if (framesSent == totalFrames) { tcp.push_back(0); tcp.push_back(0); framesSent++; break; }
            tcp.push_back(0); tcp.push_back(60);
            for (int k = 0; k < 60; k++) tcp.push_back((uint8_t)k);
            framesSent++;
        }
        // (a) intake
        bool tookBytes = false;
        if (!streamDone)
        {
            int frames = 0;
            while (frames < 16 && ring.freeSpace() >= FRAME && !tcp.empty())
            {
                size_t n = rx.wanted() < tcp.size() ? rx.wanted() : tcp.size();
                for (size_t k = 0; k < n; k++) { rx.dst()[k] = tcp.front(); tcp.pop_front(); }
                tookBytes = true;
                lastProgress = now;
                OpusRxEvent ev = rx.feed(n);
                if (ev == OPUS_RX_END) { streamDone = true; break; }
                if (ev == OPUS_RX_FRAME)
                {
                    int16_t dec[FRAME];
                    for (size_t k = 0; k < FRAME; k++) dec[k] = nextSample++;
                    ring.push(dec, FRAME);
                    frames++;
                }
            }
        }
        // (b) play one chunk
        bool wasStarted = gate.started;
        if (!wasStarted && gate.mayPlay(ring.count, thresholdSamples, streamDone)) playStart = now;
        bool started = gate.started || gate.mayPlay(ring.count, thresholdSamples, streamDone);
        if (started && !streamDone && ring.count == 0) res.underruns++;
        if (started && now - playStart >= warmMs && !streamDone && ring.count < res.minCushionAfter)
            res.minCushionAfter = ring.count;
        size_t played = 0;
        if (started && ring.count > 0)
        {
            int16_t o[CHUNK];
            played = ring.pop(o, CHUNK);
            for (size_t k = 0; k < played; k++) res.inOrder &= (o[k] == wantSample++);
            res.played += played;
        }
        if (played > 0) lastProgress = now;
        else if (streamDone && ring.count == 0) break;
        else if (!streamDone && now - lastProgress > STALL_MS) { res.aborted = true; streamDone = true; }
        (void)tookBytes;
    }
    return res;
}

static void test_simulation()
{
    const size_t S = 16000;
    const size_t FRAMES = 15 * 60 * 50; // 15 minutes of 20 ms frames

    // Whole 15-minute file, 30 s ring, a 10 s TCP outage at minute 6.
    SimResult a = simulate(30 * S, FRAMES, 6 * 60 * 1000, 6 * 60 * 1000 + 10000, 40000);
    CHECK("sim: 15 min, 10 s outage with a 30 s ring: no underrun", a.underruns == 0);
    CHECK("sim: ...not aborted by the stall timer", !a.aborted);
    CHECK("sim: ...every sample played in order", a.inOrder && a.played == FRAMES * 320);
    CHECK("sim: ...cushion stays near full after warm-up (rolling refill)",
          a.minCushionAfter > 20 * S);

    // Same outage, 1 s ring: the cushion is gone after 1 s, playback underruns, but the
    // stream resumes (outage 3 s < 5 s stall limit) and nothing is lost or reordered.
    SimResult b = simulate(1 * S, 20000, 60000, 63000, 4000);
    CHECK("sim: 1 s ring, 3 s outage: underruns", b.underruns > 0);
    CHECK("sim: ...resumes, not aborted, all samples in order", !b.aborted && b.inOrder && b.played == 20000 * 320);

    // An outage longer than the stall limit with an EMPTY ring aborts.
    SimResult c = simulate(1 * S, 20000, 60000, 90000, 4000);
    CHECK("sim: 1 s ring, 30 s outage: stall timer aborts", c.aborted);
    CHECK("sim: ...samples before the abort were played in order", c.inOrder && c.played > 0);

    // Short stream (shorter than the threshold) is played completely at stream end.
    SimResult d = simulate(30 * S, 100, 0, 0, 0); // 2 s of audio, 30 s threshold
    CHECK("sim: stream shorter than the threshold plays in full", d.inOrder && d.played == 100 * 320 && !d.aborted);
}

int main()
{
    test_ring();
    test_parser();
    test_gate();
    test_simulation();
    printf("\n%d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}

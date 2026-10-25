// play_ring.h - the rolling playback buffer of onjuino.ino, kept free of Arduino
// and ESP-IDF headers so tests/test_play_ring.cpp can compile and run it on the
// host. (20261025)
//
// PlayRing  : FIFO of 16-bit samples (volume shift is applied when a sample is
//             popped for playback, which halves the PSRAM use vs int32).
// OpusFrameRx: incremental parser for the pod's TCP framing (2-byte big-endian
//             length + Opus packet, length 0 = end-of-speech). It never blocks:
//             the caller reads whatever TCP has into dst() and calls feed().
// PlayGate  : when playback may start (threshold reached, or stream finished).
#pragma once
#include <stddef.h>
#include <stdint.h>

struct PlayRing
{
    int16_t *buf = nullptr;
    size_t cap = 0;
    size_t head = 0;  // index of the oldest sample
    size_t count = 0; // samples queued

    void begin(int16_t *b, size_t c)
    {
        buf = b;
        cap = c;
        head = 0;
        count = 0;
    }
    void clear()
    {
        head = 0;
        count = 0;
    }
    size_t freeSpace() const { return cap - count; }

    // Append up to n samples; returns how many fit.
    size_t push(const int16_t *src, size_t n)
    {
        if (n > freeSpace())
        {
            n = freeSpace();
        }
        size_t tail = head + count;
        if (tail >= cap)
        {
            tail -= cap;
        }
        for (size_t i = 0; i < n; i++)
        {
            buf[tail] = src[i];
            if (++tail == cap)
            {
                tail = 0;
            }
        }
        count += n;
        return n;
    }

    // Remove up to n samples (oldest first) into dst; returns how many.
    size_t pop(int16_t *dst, size_t n)
    {
        if (n > count)
        {
            n = count;
        }
        for (size_t i = 0; i < n; i++)
        {
            dst[i] = buf[head];
            if (++head == cap)
            {
                head = 0;
            }
        }
        count -= n;
        return n;
    }
};

enum OpusRxEvent
{
    OPUS_RX_MORE,    // need more bytes
    OPUS_RX_FRAME,   // a whole packet is in packet[0..frameLen)
    OPUS_RX_END,     // end-of-speech marker (length 0)
    OPUS_RX_BAD_LEN, // length above maxPacket: stream out of sync
};

struct OpusFrameRx
{
    uint8_t *packet = nullptr;
    size_t maxPacket = 0;
    uint8_t hdr[2] = {0, 0};
    size_t hdrGot = 0;
    size_t frameLen = 0;
    size_t got = 0;
    bool inFrame = false;

    void begin(uint8_t *p, size_t m)
    {
        packet = p;
        maxPacket = m;
        hdrGot = frameLen = got = 0;
        inFrame = false;
    }
    // Where the next received bytes go, and how many are wanted at most.
    uint8_t *dst() { return inFrame ? packet + got : hdr + hdrGot; }
    size_t wanted() const { return inFrame ? frameLen - got : 2 - hdrGot; }
    // True while a length or packet is half received (stream cut if it ends here).
    bool midFrame() const { return inFrame || hdrGot > 0; }

    // n bytes were written at dst() (n <= wanted()).
    OpusRxEvent feed(size_t n)
    {
        if (!inFrame)
        {
            hdrGot += n;
            if (hdrGot < 2)
            {
                return OPUS_RX_MORE;
            }
            frameLen = ((size_t)hdr[0] << 8) | hdr[1];
            hdrGot = 0;
            if (frameLen == 0)
            {
                return OPUS_RX_END;
            }
            if (frameLen > maxPacket)
            {
                return OPUS_RX_BAD_LEN;
            }
            inFrame = true;
            got = 0;
            return OPUS_RX_MORE;
        }
        got += n;
        if (got < frameLen)
        {
            return OPUS_RX_MORE;
        }
        inFrame = false;
        got = 0;
        return OPUS_RX_FRAME;
    }
};

struct PlayGate
{
    bool started = false;

    // True when a chunk may be played now. Playback starts once `count` reaches
    // `threshold`, or when the stream is over (a short stream below the
    // threshold is still played in full). Once started it stays started: an
    // empty ring (underrun) is not re-buffered, it just waits for more data.
    bool mayPlay(size_t count, size_t threshold, bool streamDone)
    {
        if (!started && count > 0 && (count >= threshold || streamDone))
        {
            started = true;
        }
        return started && count > 0;
    }
};

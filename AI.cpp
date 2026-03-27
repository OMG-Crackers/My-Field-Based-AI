/*
 * UNIFIED QUANTUM FIELD AI — Complete Engine
 *
 * ONE file. ONE field. ALL capabilities.
 * 48 dimensions. 3 primitives: deposit, overlap, direction.
 *
 * VISION:    see pixels → extract features → dims 0-9
 * PHYSICS:   lift/tap/squeeze/push → force curves → dims 10-21
 * DERIVED:   brain computes brittleness/toughness/softness → dims 22-26
 * TEXT:      classification, generation, reasoning → dims 27-30
 * CAUSAL:    cause→effect chains, backward recall → dims 31-33
 * TEMPORAL:  before/during/after trajectories → dims 34-35
 * OUTCOMES:  drop/throw/squeeze/push results → dims 36-39
 * META:      familiarity, experience, confidence → dims 40-43
 * CONTEXT:   domain separation → dims 44-47
 *
 * Compile: g++ -O3 -std=c++17 -o unified unified_field_ai.cpp -lm -lpthread
 */

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <fstream>
#include <sstream>
#include <functional>
#include <numeric>

// ═══════════════════════════════════════════════════════════════
//  QUANTUM VECTOR — 48 dimensions, all info simultaneously
// ═══════════════════════════════════════════════════════════════

constexpr int QDIMS = 48;
constexpr int IMG_W = 32, IMG_H = 32;

struct QVec {
    float d[QDIMS] = {};

    QVec() { memset(d, 0, sizeof(d)); }

    float dist_sq(const QVec& o) const {
        float s = 0;
        for (int i = 0; i < QDIMS; i++) { float x = d[i] - o.d[i]; s += x * x; }
        return s;
    }

    QVec operator+(const QVec& o) const {
        QVec r; for (int i = 0; i < QDIMS; i++) r.d[i] = d[i] + o.d[i]; return r;
    }
    QVec operator-(const QVec& o) const {
        QVec r; for (int i = 0; i < QDIMS; i++) r.d[i] = d[i] - o.d[i]; return r;
    }
    QVec operator*(float s) const {
        QVec r; for (int i = 0; i < QDIMS; i++) r.d[i] = d[i] * s; return r;
    }
    QVec& operator+=(const QVec& o) {
        for (int i = 0; i < QDIMS; i++) d[i] += o.d[i]; return *this;
    }
};

enum QDim {
    // VISION (from pixel processing)
    V_SIZE = 0, V_OPACITY = 1, V_HUE = 2, V_BRIGHT_VAR = 3, V_EDGES = 4,
    V_TEXTURE = 5, V_ROUNDNESS = 6, V_SYMMETRY = 7, V_TRANSPARENCY = 8, V_SHARPNESS = 9,

    // PHYSICAL (from interaction)
    P_WEIGHT = 10, P_TEMP = 11, P_FRICTION = 12, P_TEXTURE = 13,
    P_TAP_SOUND = 14, P_TAP_REBOUND = 15,
    P_SQ_RESIST = 16, P_SQ_DEFORM = 17, P_SQ_SPRING = 18, P_SQ_BROKE = 19,
    P_PUSH_FRICTION = 20, P_SLIDE_SOUND = 21,

    // DERIVED (brain computes from interaction)
    D_BRITTLE = 22, D_TOUGH = 23, D_SOFT = 24, D_SLIPPERY = 25, D_DENSITY = 26,

    // TEXT/MEANING (from language training)
    T_MEANING_X = 27, T_MEANING_Y = 28, T_WEIGHT = 29, T_CONCEPT = 30,

    // CAUSAL (from observed cause->effect)
    C_CAUSE_ROLE = 31, C_EFFECT_ROLE = 32, C_STRENGTH = 33,

    // TEMPORAL
    TEMP_PHASE = 34, TEMP_VALENCE = 35,

    // OUTCOMES (running averages)
    O_DROP = 36, O_THROW = 37, O_SQUEEZE = 38, O_PUSH = 39,

    // META
    M_FAMILIARITY = 40, M_EXPERIENCE = 41, M_CONFIDENCE = 42, M_SALIENCE = 43,

    // CONTEXT — separates domains so they don't bleed into each other
    // Each entity accumulates context from what it co-occurs with.
    // "rain" and "wet_ground" share context because they were observed together.
    // "rain" and "fire" have different contexts because they never co-occur.
    // This prevents cross-domain contamination in the shared field.
    CTX_A = 44, CTX_B = 45, CTX_C = 46, CTX_D = 47,
};

// ═══════════════════════════════════════════════════════════════
//  PIXEL IMAGE + COMPUTER VISION
// ═══════════════════════════════════════════════════════════════

struct Pixel { float r, g, b; };

struct Image {
    Pixel px[IMG_H][IMG_W];

    void clear(float r, float g, float b) {
        for (int y = 0; y < IMG_H; y++)
            for (int x = 0; x < IMG_W; x++)
                px[y][x] = {r, g, b};
    }

    void circle(int cx, int cy, int rad, float r, float g, float b, float a = 1) {
        for (int y = 0; y < IMG_H; y++)
            for (int x = 0; x < IMG_W; x++) {
                float dx = x - cx, dy = y - cy;
                if (dx * dx + dy * dy <= rad * rad) {
                    px[y][x].r = px[y][x].r * (1 - a) + r * a;
                    px[y][x].g = px[y][x].g * (1 - a) + g * a;
                    px[y][x].b = px[y][x].b * (1 - a) + b * a;
                }
            }
    }

    void rect(int x0, int y0, int x1, int y1, float r, float g, float b, float a = 1) {
        for (int y = std::max(0, y0); y < std::min(IMG_H, y1); y++)
            for (int x = std::max(0, x0); x < std::min(IMG_W, x1); x++) {
                px[y][x].r = px[y][x].r * (1 - a) + r * a;
                px[y][x].g = px[y][x].g * (1 - a) + g * a;
                px[y][x].b = px[y][x].b * (1 - a) + b * a;
            }
    }

    void highlight(int cx, int cy, int rad, float intensity) {
        for (int y = 0; y < IMG_H; y++)
            for (int x = 0; x < IMG_W; x++) {
                float dx = x - cx, dy = y - cy, d2 = dx * dx + dy * dy;
                if (d2 < rad * rad) {
                    float f = 1 - sqrtf(d2) / (float)rad;
                    float b = f * f * intensity;
                    px[y][x].r = std::min(1.0f, px[y][x].r + b);
                    px[y][x].g = std::min(1.0f, px[y][x].g + b);
                    px[y][x].b = std::min(1.0f, px[y][x].b + b);
                }
            }
    }

    void noise(std::mt19937& rng, float amt) {
        std::normal_distribution<float> n(0, amt);
        for (int y = 0; y < IMG_H; y++)
            for (int x = 0; x < IMG_W; x++) {
                px[y][x].r = std::clamp(px[y][x].r + n(rng), 0.f, 1.f);
                px[y][x].g = std::clamp(px[y][x].g + n(rng), 0.f, 1.f);
                px[y][x].b = std::clamp(px[y][x].b + n(rng), 0.f, 1.f);
            }
    }
};

static float rgb_hue(float r, float g, float b) {
    float mx = std::max({r, g, b}), mn = std::min({r, g, b});
    if (mx - mn < 0.001f) return 0;
    float h;
    if (mx == r)      h = (g - b) / (mx - mn);
    else if (mx == g)  h = 2 + (b - r) / (mx - mn);
    else               h = 4 + (r - g) / (mx - mn);
    h /= 6;
    if (h < 0) h += 1;
    return h;
}

static void extract_vision(const Image& img, float bg_r, float bg_g, float bg_b, QVec& out) {
    float thr = 0.15f;
    int cnt = 0;
    float sr = 0, sg = 0, sb = 0, sbr = 0;
    float mnx = IMG_W, mxx = 0, mny = IMG_H, mxy = 0;

    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++) {
            auto& p = img.px[y][x];
            float dr = p.r - bg_r, dg = p.g - bg_g, db = p.b - bg_b;
            if (sqrtf(dr * dr + dg * dg + db * db) > thr) {
                cnt++; sr += p.r; sg += p.g; sb += p.b;
                sbr += (p.r + p.g + p.b) / 3;
                if (x < mnx) mnx = x; if (x > mxx) mxx = x;
                if (y < mny) mny = y; if (y > mxy) mxy = y;
            }
        }
    if (cnt < 5) return;

    float ar = sr / cnt, ag = sg / cnt, ab = sb / cnt, abr = sbr / cnt;
    out.d[V_HUE] = rgb_hue(ar, ag, ab);
    float mx = std::max({ar, ag, ab}), mn = std::min({ar, ag, ab});
    out.d[V_OPACITY] = mx > 0.01f ? (mx - mn) / mx : 0;
    out.d[V_SIZE] = (float)cnt / (IMG_W * IMG_H);

    // Brightness variance
    float bv = 0;
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++) {
            auto& p = img.px[y][x];
            float dr = p.r - bg_r, dg = p.g - bg_g, db = p.b - bg_b;
            if (sqrtf(dr * dr + dg * dg + db * db) > thr) {
                float b = (p.r + p.g + p.b) / 3 - abr;
                bv += b * b;
            }
        }
    out.d[V_BRIGHT_VAR] = sqrtf(bv / cnt);

    // Sobel edges
    float esum = 0; int ecnt = 0;
    for (int y = 1; y < IMG_H - 1; y++)
        for (int x = 1; x < IMG_W - 1; x++) {
            float gx = -img.px[y-1][x-1].r - 2*img.px[y][x-1].r - img.px[y+1][x-1].r
                        + img.px[y-1][x+1].r + 2*img.px[y][x+1].r + img.px[y+1][x+1].r;
            float gy = -img.px[y-1][x-1].r - 2*img.px[y-1][x].r - img.px[y-1][x+1].r
                        + img.px[y+1][x-1].r + 2*img.px[y+1][x].r + img.px[y+1][x+1].r;
            esum += sqrtf(gx * gx + gy * gy); ecnt++;
        }
    out.d[V_EDGES] = std::min(ecnt > 0 ? esum / ecnt * 3 : 0.f, 1.f);

    // Texture (patch variance)
    float pvs = 0; int np = 0;
    for (int py = 0; py < IMG_H - 4; py += 4)
        for (int px = 0; px < IMG_W - 4; px += 4) {
            float pm = 0, pv = 0;
            for (int dy = 0; dy < 4; dy++)
                for (int dx = 0; dx < 4; dx++)
                    pm += (img.px[py+dy][px+dx].r + img.px[py+dy][px+dx].g + img.px[py+dy][px+dx].b) / 3;
            pm /= 16;
            for (int dy = 0; dy < 4; dy++)
                for (int dx = 0; dx < 4; dx++) {
                    float b = (img.px[py+dy][px+dx].r + img.px[py+dy][px+dx].g + img.px[py+dy][px+dx].b) / 3 - pm;
                    pv += b * b;
                }
            pvs += pv / 16; np++;
        }
    out.d[V_TEXTURE] = np > 0 ? std::min(sqrtf(pvs / np) * 5, 1.f) : 0;

    // Roundness
    float bw = mxx - mnx + 1, bh = mxy - mny + 1, ba = bw * bh;
    out.d[V_ROUNDNESS] = ba > 0 ? 1 - (float)cnt / ba * 0.5f : 0;

    // Symmetry
    float sd = 0; int sc = 0; int cx2 = (mnx + mxx) / 2;
    for (int y = (int)mny; y <= (int)mxy; y++)
        for (int dx = 1; dx < (int)(bw / 2); dx++) {
            int lx = cx2 - dx, rx = cx2 + dx;
            if (lx >= 0 && rx < IMG_W) {
                float dr = img.px[y][lx].r - img.px[y][rx].r;
                float dg2 = img.px[y][lx].g - img.px[y][rx].g;
                float db2 = img.px[y][lx].b - img.px[y][rx].b;
                sd += sqrtf(dr * dr + dg2 * dg2 + db2 * db2); sc++;
            }
        }
    out.d[V_SYMMETRY] = sc > 0 ? std::max(0.f, 1 - sd / sc * 3) : 0;

    // Transparency
    int semi = 0;
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++) {
            auto& p = img.px[y][x];
            float dr = p.r - bg_r, dg = p.g - bg_g, db = p.b - bg_b;
            float d = sqrtf(dr * dr + dg * dg + db * db);
            if (d > 0.05f && d < 0.3f) semi++;
        }
    out.d[V_TRANSPARENCY] = std::min((float)semi / (cnt + 1) * 2, 1.f);

    // Sharpness
    int se = 0, we = 0;
    for (int y = 1; y < IMG_H - 1; y++)
        for (int x = 1; x < IMG_W - 1; x++) {
            float gx2 = img.px[y][x+1].r - img.px[y][x-1].r;
            float gy2 = img.px[y+1][x].r - img.px[y-1][x].r;
            float m = sqrtf(gx2 * gx2 + gy2 * gy2);
            if (m > 0.3f) se++; else if (m > 0.05f) we++;
        }
    out.d[V_SHARPNESS] = (se + we) > 0 ? (float)se / (se + we) : 0;
}

// ═══════════════════════════════════════════════════════════════
//  PHYSICAL OBJECT (for embodied simulation)
// ═══════════════════════════════════════════════════════════════

struct PhysObj {
    std::string name, material;
    float color_r, color_g, color_b, shininess, tex_freq, transparency;
    bool is_round;
    int render_size;
    float density, yield_mpa, ult_mpa, youngs_gpa, elastic_limit, wall_mm;
    float friction, roughness, sound_hz, cor, temperature;

    float fragility() const {
        float t = ult_mpa * elastic_limit * wall_mm;
        float tn = t / 5;
        float s = youngs_gpa / (youngs_gpa + 1);
        return std::clamp((1 - tn) * s, 0.f, 1.f);
    }

    Image render(std::mt19937& rng) const {
        Image img;
        img.clear(0.4f, 0.4f, 0.45f);
        int cx = IMG_W / 2, cy = IMG_H / 2;
        float a = 1 - transparency * 0.7f;
        if (is_round)
            img.circle(cx, cy, render_size, color_r, color_g, color_b, a);
        else
            img.rect(cx - render_size, cy - render_size * 2 / 3,
                     cx + render_size, cy + render_size * 2 / 3,
                     color_r, color_g, color_b, a);
        if (shininess > 0.3f)
            img.highlight(cx - render_size / 3, cy - render_size / 3, render_size / 2, shininess);
        img.noise(rng, 0.03f);
        return img;
    }
};

// ═══════════════════════════════════════════════════════════════
//  HELPER
// ═══════════════════════════════════════════════════════════════

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::string t;
    for (char c : s) {
        if (c == ' ' || c == '\t' || c == '\n') {
            if (!t.empty()) { out.push_back(t); t.clear(); }
        } else {
            t += tolower(c);
        }
    }
    if (!t.empty()) out.push_back(t);
    return out;
}

// ═══════════════════════════════════════════════════════════════
//  STRUCTURES
// ═══════════════════════════════════════════════════════════════

struct Deposit    { QVec pos; float weight; int ts; };
struct Trajectory { std::string action, target; float outcome; int ts; };
struct CausalLink { std::string cause, effect; float strength; int ts; };
struct ConvTurn   { std::string speaker, text; int ts; };

// ═══════════════════════════════════════════════════════════════
//  THE UNIFIED FIELD AI
// ═══════════════════════════════════════════════════════════════

class UnifiedAI {
public:
    // ───── Core parameters ─────
    float sigma = 0.15f, rbf_denom;
    int time_step = 0;

    // ───── THE FIELD: one vector per known entity ─────
    std::unordered_map<std::string, QVec> positions;

    // ───── Deposits (decaying memory) ─────
    std::vector<Deposit> memory_field, meta_field;

    // ───── Text classification ─────
    struct ConceptInfo { QVec center; std::vector<std::string> words; float total_wt; };
    std::unordered_map<std::string, ConceptInfo> concepts;
    std::vector<std::string> concept_names;
    std::unordered_map<std::string, std::unordered_map<std::string, float>> word_cprobs;
    std::unordered_map<std::string, QVec> avg_next;
    std::vector<std::pair<std::string, std::vector<std::string>>> training_data;

    // ───── Physical ─────
    std::vector<Trajectory> trajectories;
    std::unordered_map<std::string, std::unordered_set<std::string>> tried;
    std::unordered_map<std::string, std::vector<std::string>> obj_features;

    // ───── Causal ─────
    std::vector<CausalLink> causal_links;
    std::unordered_map<std::string, std::vector<std::pair<std::string, QVec>>> causal_fwd, causal_rev;
    std::unordered_map<std::string, std::unordered_set<std::string>> causal_siblings;

    // ───── Conversation ─────
    std::vector<ConvTurn> conv_history;

    // ───── Autonomous ─────
    std::atomic<bool> alive{false};
    std::mutex input_mutex;
    std::vector<std::vector<std::string>> pending_inputs;
    float energy = 1, boredom = 0;
    int auto_tick = 0, last_resolve = 0;

    // ───── Stop words ─────
    std::unordered_set<std::string> stops;

    // ═══════════════════════════════════════
    //  CONSTRUCTOR
    // ═══════════════════════════════════════

    UnifiedAI() {
        rbf_denom = 4 * sigma * sigma;
        const char* sl[] = {
            "the","a","an","in","of","to","for","by","on","at","from","with",
            "and","was","were","is","are","has","been","after","their","this","that","more",
            "about","into","also","new","it","its","but","not","or","than","other","some",
            "will","can","would","should","had","have","do","does","did", nullptr
        };
        for (int i = 0; sl[i]; i++) stops.insert(sl[i]);
    }

    // ═══════════════════════════════════════
    //  THREE PRIMITIVES
    // ═══════════════════════════════════════

    // Primitive 1: OVERLAP — how similar are two field positions?
    float overlap(const QVec& a, const QVec& b) const {
        return expf(-a.dist_sq(b) / rbf_denom);
    }

    // Channel-specific overlap (compare only dims s..e)
    float overlap_ch(const QVec& a, const QVec& b, int s, int e) const {
        float sum = 0;
        for (int i = s; i <= e; i++) { float x = a.d[i] - b.d[i]; sum += x * x; }
        return expf(-sum / (rbf_denom * 2));
    }

    // Weighted multi-channel overlap with data-driven gating
    float smart_overlap(const QVec& a, const QVec& b) const {
        struct Ch { int s, e; float w; };
        Ch chs[] = {
            {V_SIZE,       V_SHARPNESS,   1.5f},
            {P_WEIGHT,     P_SLIDE_SOUND, 2.0f},
            {D_BRITTLE,    D_DENSITY,     2.5f},
            {T_MEANING_X,  T_CONCEPT,     1.0f},
            {O_DROP,       O_PUSH,        2.0f},
            {M_FAMILIARITY,M_SALIENCE,    0.3f}
        };
        float ts = 0, tw = 0;
        for (auto& c : chs) {
            float ae = 0, be = 0;
            for (int d = c.s; d <= c.e; d++) { ae += fabsf(a.d[d]); be += fabsf(b.d[d]); }
            if (ae < 0.01f && be < 0.01f) continue;
            float sim = overlap_ch(a, b, c.s, c.e);
            float dw = std::min(ae, be) / (std::max(ae, be) + 0.01f);
            float w = c.w * (0.3f + 0.7f * dw);
            ts += sim * w; tw += w;
        }
        return tw > 0 ? ts / tw : 0;
    }

    // Primitive 2: DIRECTION — vector from a to b in the field
    QVec direction(const QVec& a, const QVec& b) const { return b - a; }

    // Follow a direction from a position
    QVec follow(const QVec& a, const QVec& d) const { return a + d; }

    // Primitive 3: DEPOSIT — place weighted memory into the field
    void deposit(std::vector<Deposit>& f, const QVec& p, float w) {
        time_step++;
        f.push_back({p, w, time_step});
        if (f.size() > 1500) f.erase(f.begin(), f.begin() + 750);
    }

    // Read field energy at a position (sum of decayed deposits weighted by overlap)
    float field_energy(const std::vector<Deposit>& f, const QVec& p, float decay = 0.02f) const {
        float t = 0;
        for (auto& d : f) {
            float a = time_step - d.ts;
            t += d.weight * expf(-decay * a) * overlap(p, d.pos);
        }
        return t;
    }

    // Find the strongest recent deposit
    std::pair<QVec, float> field_peak(const std::vector<Deposit>& f, float decay = 0.03f) const {
        QVec best; float be = 0;
        int s = std::max(0, (int)f.size() - 50);
        for (int i = s; i < (int)f.size(); i++) {
            float a = time_step - f[i].ts;
            float e = f[i].weight * expf(-decay * a);
            if (e > be) { be = e; best = f[i].pos; }
        }
        return {best, be};
    }

    // ═══════════════════════════════════════
    //  ENTITY MANAGEMENT
    // ═══════════════════════════════════════

    void ensure(const std::string& w) {
        if (positions.count(w)) return;
        QVec p;
        std::mt19937 rng(std::hash<std::string>{}(w));
        std::normal_distribution<float> nd(0, 0.3f);
        p.d[T_MEANING_X] = nd(rng);
        p.d[T_MEANING_Y] = nd(rng);
        // Each entity starts with unique context fingerprint
        p.d[CTX_A] = nd(rng);
        p.d[CTX_B] = nd(rng);
        p.d[CTX_C] = nd(rng);
        p.d[CTX_D] = nd(rng);
        positions[w] = p;
    }

    bool is_stop(const std::string& w) const { return stops.count(w); }

    std::pair<std::string, float> nearest(const QVec& p,
            const std::unordered_set<std::string>& exc = {}, bool skip_stops = false) const {
        std::string best; float bov = 0;
        for (auto& [w, pos] : positions) {
            if (exc.count(w)) continue;
            if (skip_stops && stops.count(w)) continue;
            float ov = overlap_ch(p, pos, T_MEANING_X, T_CONCEPT);
            if (ov > bov) { bov = ov; best = w; }
        }
        return {best, bov};
    }

    // ═══════════════════════════════════════
    //  TEXT: TRAIN
    // ═══════════════════════════════════════

    void train(const std::vector<std::pair<std::string, std::vector<std::string>>>& data) {
        training_data = data;

        // Count word-concept co-occurrences
        std::unordered_map<std::string, std::unordered_map<std::string, int>> wcc;
        std::unordered_set<std::string> cseen;
        concept_names.clear();
        for (auto& [name, tokens] : data) {
            if (!cseen.count(name)) { cseen.insert(name); concept_names.push_back(name); }
            std::unordered_set<std::string> seen;
            for (auto& t : tokens)
                if (!seen.count(t)) { seen.insert(t); wcc[t][name]++; }
        }

        // Place concepts evenly around a circle in meaning space
        int nc = concept_names.size();
        concepts.clear();
        for (int i = 0; i < nc; i++) {
            ConceptInfo ci; ci.center = QVec(); ci.total_wt = 0;
            float angle = 2 * M_PI * i / nc;
            ci.center.d[T_MEANING_X] = cosf(angle) * 2;
            ci.center.d[T_MEANING_Y] = sinf(angle) * 2;
            ci.center.d[T_CONCEPT] = (float)i / nc;
            concepts[concept_names[i]] = ci;
        }

        // Position each word near its dominant concept
        word_cprobs.clear();
        for (auto& [word, cc] : wcc) {
            int total = 0; std::string mx_c; int mx_n = 0;
            for (auto& [cn, cnt] : cc) { total += cnt; if (cnt > mx_n) { mx_n = cnt; mx_c = cn; } }
            float conc = (float)mx_n / total;
            float wt = conc > 0.5f ? conc * 1.2f : (conc > 0.3f ? conc : 0.f);
            ensure(word);
            if (conc >= 0.3f) {
                auto& center = concepts[mx_c].center;
                std::mt19937 rng(std::hash<std::string>{}(word));
                std::normal_distribution<float> nd(0, std::max(0.1f, (1 - conc) * 0.5f));
                positions[word].d[T_MEANING_X] = center.d[T_MEANING_X] + nd(rng);
                positions[word].d[T_MEANING_Y] = center.d[T_MEANING_Y] + nd(rng);
                positions[word].d[T_WEIGHT] = wt;
                positions[word].d[T_CONCEPT] = center.d[T_CONCEPT] + nd(rng) * 0.1f;
            }
            for (auto& [cn, cnt] : cc)
                word_cprobs[word][cn] = (float)cnt / total;
        }

        // Build concept word lists
        for (auto& [cn, ci] : concepts) { ci.words.clear(); ci.total_wt = 0; }
        for (auto& [word, probs] : word_cprobs) {
            float wt = positions[word].d[T_WEIGHT];
            if (wt < 0.01f) continue;
            for (auto& [cn, ci] : concepts)
                if (probs.count(cn) && probs[cn] > 0.1f) {
                    ci.words.push_back(word);
                    ci.total_wt += wt;
                }
        }

        // Directions for generation (average next-word vectors)
        avg_next.clear();
        std::unordered_map<std::string, std::vector<QVec>> dirs;
        for (auto& [name, tokens] : data) {
            std::vector<std::string> ct;
            for (auto& t : tokens)
                if (positions[t].d[T_WEIGHT] > 0.01f && !is_stop(t))
                    ct.push_back(t);
            for (int i = 0; i < (int)ct.size() - 1; i++)
                dirs[ct[i]].push_back(direction(positions[ct[i]], positions[ct[i+1]]));
        }
        for (auto& [w, ds] : dirs) {
            QVec a;
            for (auto& d : ds) a += d;
            avg_next[w] = a * (1.f / ds.size());
        }
    }

    // Incremental learning: add new example and retrain
    void learn(const std::string& name, const std::vector<std::string>& tokens) {
        training_data.push_back({name, tokens});
        train(training_data);
    }

    // ═══════════════════════════════════════
    //  TEXT: CLASSIFY
    // ═══════════════════════════════════════

    struct Scores {
        std::unordered_map<std::string, float> all;
        std::string best;
        float confidence = 0;
    };

    Scores classify(const std::vector<std::string>& tokens) {
        struct QW { std::string w; float wt; };
        std::vector<QW> query;
        for (auto& t : tokens)
            if (positions.count(t) && positions[t].d[T_WEIGHT] > 0.01f)
                query.push_back({t, positions[t].d[T_WEIGHT]});

        Scores res;
        if (query.empty()) return res;

        int nc = concept_names.size();
        std::vector<float> rbf_s(nc, 0), vote_s(nc, 0);

        for (int ci = 0; ci < nc; ci++) {
            auto& cn = concept_names[ci];
            auto& cc = concepts[cn];
            if (cc.total_wt < 1e-10f) continue;
            float tot = 0;
            for (auto& q : query)
                for (auto& cw : cc.words)
                    tot += q.wt * positions[cw].d[T_WEIGHT] *
                           overlap_ch(positions[q.w], positions[cw], T_MEANING_X, T_CONCEPT);
            rbf_s[ci] = tot / cc.total_wt;
            for (auto& q : query) {
                auto& p = word_cprobs[q.w];
                if (p.count(cn)) vote_s[ci] += p[cn] * q.wt;
            }
        }

        float rm = *std::max_element(rbf_s.begin(), rbf_s.end());
        float vm = *std::max_element(vote_s.begin(), vote_s.end());
        if (rm > 0) for (auto& s : rbf_s) s /= rm;
        if (vm > 0) for (auto& s : vote_s) s /= vm;

        std::vector<float> raw(nc);
        for (int i = 0; i < nc; i++)
            raw[i] = (rbf_s[i] * 0.5f + vote_s[i] * 0.5f) * 5;

        float mx2 = *std::max_element(raw.begin(), raw.end());
        float sum = 0;
        for (auto& r : raw) { r = expf(r - mx2); sum += r; }

        for (int i = 0; i < nc; i++) {
            float s = raw[i] / sum;
            res.all[concept_names[i]] = s;
            if (s > res.confidence) { res.confidence = s; res.best = concept_names[i]; }
        }
        return res;
    }

    // ═══════════════════════════════════════
    //  TEXT: THINK, REASON, GENERATE, CHAT
    // ═══════════════════════════════════════

    struct Thought {
        std::string answer;
        float confidence;
        std::vector<std::string> thoughts;
    };

    Thought think(const std::vector<std::string>& tokens, bool verbose = true) {
        time_step++;
        Thought th;
        auto sc = classify(tokens);
        th.answer = sc.best;
        th.confidence = sc.confidence;

        if (!sc.best.empty() && sc.confidence > 0.01f)
            th.thoughts.push_back("I see: " + sc.best);
        else
            th.thoughts.push_back("I don't recognize this.");

        for (auto& t : tokens)
            if (!positions.count(t) && !is_stop(t)) {
                th.thoughts.push_back("New: " + t);
                break;
            }

        // Deposit uncertainty into meta field for self-repair
        if (!sc.best.empty() && sc.confidence < 0.8f && concepts.count(sc.best))
            deposit(meta_field, concepts[sc.best].center, 1 - sc.confidence);

        if (verbose)
            for (auto& t : th.thoughts) printf("  %s\n", t.c_str());
        return th;
    }

    struct Chain { std::vector<std::string> words; };

    Chain reason(const std::vector<std::string>& seed, int steps = 3) {
        Chain ch;
        ch.words = seed;
        std::vector<std::string> ct;
        for (auto& t : seed)
            if (positions.count(t) && !is_stop(t)) ct.push_back(t);
        std::string cur = ct.empty() ? (seed.empty() ? "" : seed.back()) : ct.back();
        std::unordered_set<std::string> used(ch.words.begin(), ch.words.end());

        for (int i = 0; i < steps; i++) {
            if (!avg_next.count(cur) || !positions.count(cur)) break;
            QVec pred = follow(positions[cur], avg_next[cur]);
            auto [nw, ov] = nearest(pred, used, true);
            if (nw.empty()) break;
            ch.words.push_back(nw);
            used.insert(nw);
            cur = nw;
        }
        return ch;
    }

    std::vector<std::string> generate(const std::vector<std::string>& seed, int max_len = 8) {
        auto out = seed;
        auto sc = classify(seed);
        QVec topic;
        if (!sc.best.empty() && concepts.count(sc.best))
            topic = concepts[sc.best].center;
        std::unordered_set<std::string> used(out.begin(), out.end());

        for (int step = 0; step < max_len - (int)seed.size(); step++) {
            auto& cur = out.back();
            if (!positions.count(cur) || !avg_next.count(cur)) break;
            QVec pred = follow(positions[cur], avg_next[cur]);
            std::string best; float bs = -1;

            for (auto& [w, p] : positions) {
                if (p.d[T_WEIGHT] < 0.01f || is_stop(w)) continue;
                float d_ov = overlap_ch(pred, p, T_MEANING_X, T_CONCEPT);
                float t_ov = overlap_ch(p, topic, T_MEANING_X, T_CONCEPT);
                float div = used.count(w) ? 0.2f : 1.f;
                float s = (d_ov * 0.4f + t_ov * 0.6f) * div;
                if (s > bs) { bs = s; best = w; }
            }
            if (best.empty()) break;
            out.push_back(best);
            used.insert(best);
        }
        return out;
    }

    std::string analogy(const std::string& a, const std::string& b, const std::string& c) {
        if (!positions.count(a) || !positions.count(b) || !positions.count(c)) return "?";
        QVec d = direction(positions[a], positions[b]);
        QVec pred = follow(positions[c], d);
        auto [w, ov] = nearest(pred, {a, b, c}, true);
        return w;
    }

    std::string chat(const std::string& text) {
        auto tokens = split(text);
        auto th = think(tokens, false);
        std::string resp;

        std::vector<std::string> unk;
        for (auto& t : tokens)
            if (!positions.count(t) && !is_stop(t)) unk.push_back(t);

        auto ct = [&] {
            std::vector<std::string> r;
            for (auto& t : tokens)
                if (positions.count(t) && positions[t].d[T_WEIGHT] > 0.01f && !is_stop(t))
                    r.push_back(t);
            return r;
        }();

        if (!unk.empty() && (int)unk.size() > (int)ct.size())
            resp = "i do not know " + unk[0];
        else if (th.confidence < 0.3f)
            resp = "i am not sure what that means";
        else {
            resp = "that is " + th.answer;
            if (!ct.empty()) {
                auto g = generate({ct[0]}, 4);
                if (g.size() > 1) {
                    resp += ".";
                    for (int i = 1; i < (int)g.size(); i++) resp += " " + g[i];
                }
            }
        }

        conv_history.push_back({"user", text, time_step});
        conv_history.push_back({"ai", resp, time_step});
        if (conv_history.size() > 100)
            conv_history.erase(conv_history.begin(), conv_history.begin() + 50);
        return resp;
    }

    std::string chat_history(int n = 5) {
        std::string out;
        int s = std::max(0, (int)conv_history.size() - n * 2);
        for (int i = s; i < (int)conv_history.size(); i++)
            out += std::string("  ") +
                   (conv_history[i].speaker == "user" ? "You" : "AI ") + ": " +
                   conv_history[i].text + "\n";
        return out.empty() ? "No conversation yet." : out;
    }

    // ═══════════════════════════════════════
    //  TEXT: MEMORY, GOALS, NOVELTY, EVALUATE
    // ═══════════════════════════════════════

    void remember(const std::vector<std::string>& tokens) {
        for (auto& t : tokens)
            if (positions.count(t))
                deposit(memory_field, positions[t], 1);
    }

    std::vector<std::pair<std::string, float>> recall(const std::vector<std::string>& tokens, int top = 5) {
        std::vector<std::pair<std::string, float>> out;
        for (auto& t : tokens) {
            if (!positions.count(t)) continue;
            float e = field_energy(memory_field, positions[t], 0.02f);
            if (e > 0.01f) out.push_back({t, e});
        }
        std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.second > b.second; });
        if ((int)out.size() > top) out.resize(top);
        return out;
    }

    struct Goal { std::string focus; float energy; };

    std::vector<Goal> goals() {
        std::vector<Goal> out;
        std::unordered_set<std::string> seen;
        auto sorted = meta_field;
        std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.weight > b.weight; });
        for (auto& dep : sorted) {
            auto [w, _] = nearest(dep.pos, {}, true);
            if (!w.empty() && !seen.count(w)) {
                seen.insert(w);
                out.push_back({w, dep.weight});
                if (out.size() >= 5) break;
            }
        }
        return out;
    }

    struct Novelty { bool is_novel; float confidence; };

    Novelty detect_novelty(const std::vector<std::string>& tokens) {
        int known = 0, unknown = 0;
        for (auto& t : tokens) {
            if (positions.count(t) && positions[t].d[T_WEIGHT] > 0.01f) known++;
            else if (!is_stop(t)) unknown++;
        }
        auto sc = classify(tokens);
        return {known == 0 || (unknown > known && sc.confidence < 0.3f), sc.confidence};
    }

    float evaluate(const std::vector<std::string>& tokens) {
        std::vector<std::string> ct;
        for (auto& t : tokens)
            if (positions.count(t) && positions[t].d[T_WEIGHT] > 0.01f && !is_stop(t))
                ct.push_back(t);
        if (ct.size() < 2) return 0;

        float grounding = 0;
        for (auto& w : ct) grounding += std::min(positions[w].d[T_WEIGHT], 1.f);
        grounding /= ct.size();

        float consistency = 0; int pairs = 0;
        for (int i = 0; i < (int)ct.size() - 1; i++) {
            consistency += overlap_ch(positions[ct[i]], positions[ct[i+1]], T_MEANING_X, T_CONCEPT);
            pairs++;
        }
        if (pairs > 0) consistency /= pairs;

        float topicality = 0;
        QVec centroid;
        for (auto& w : ct) centroid += positions[w];
        centroid = centroid * (1.f / ct.size());
        for (auto& [cn, ci] : concepts)
            topicality = std::max(topicality, overlap_ch(centroid, ci.center, T_MEANING_X, T_CONCEPT));

        return grounding * 0.3f + consistency * 0.3f + topicality * 0.4f;
    }

    std::string imagine_blend(const std::string& c1, const std::string& c2) {
        if (!concepts.count(c1) || !concepts.count(c2)) return "unknown";
        QVec mid = (concepts[c1].center + concepts[c2].center) * 0.5f;
        std::vector<std::pair<std::string, float>> cands;
        for (auto& [w, p] : positions) {
            if (p.d[T_WEIGHT] < 0.1f || is_stop(w)) continue;
            cands.push_back({w, overlap_ch(p, mid, T_MEANING_X, T_CONCEPT)});
        }
        std::sort(cands.begin(), cands.end(), [](auto& a, auto& b) { return a.second > b.second; });
        std::string out;
        for (int i = 0; i < std::min(6, (int)cands.size()); i++) out += cands[i].first + " ";
        return out;
    }

    std::string introspect() {
        std::string out = "=== INTROSPECTION ===\n";
        out += "  Time: " + std::to_string(time_step) + "\n";
        out += "  Concepts: " + std::to_string(concept_names.size()) + "\n";
        out += "  Entities: " + std::to_string(positions.size()) + "\n";
        out += "  Memories: " + std::to_string(memory_field.size()) + "\n";
        out += "  Trajectories: " + std::to_string(trajectories.size()) + "\n";
        out += "  Causal links: " + std::to_string(causal_links.size()) + "\n";
        auto g = goals();
        if (!g.empty()) out += "  Focus: " + g[0].focus + "\n";
        return out;
    }

    // ═══════════════════════════════════════
    //  SELF-REPAIR
    // ═══════════════════════════════════════

    struct Repair { std::string action; int boosted; };

    Repair self_repair() {
        auto [pk, pe] = field_peak(meta_field);
        if (pe < 0.3f) return {"nothing", 0};
        std::string bc; float bov = 0;
        for (auto& [cn, ci] : concepts) {
            float ov = overlap(pk, ci.center);
            if (ov > bov) { bov = ov; bc = cn; }
        }
        if (bc.empty()) return {"nothing", 0};
        int boosted = 0;
        for (auto& [w, probs] : word_cprobs)
            if (probs.count(bc) && probs[bc] > 0.8f && positions.count(w)) {
                positions[w].d[T_WEIGHT] = std::min(positions[w].d[T_WEIGHT] * 1.1f, 2.f);
                boosted++;
            }
        return {"boosted " + bc, boosted};
    }

    // ═══════════════════════════════════════
    //  VISION: see rendered object
    // ═══════════════════════════════════════

    void see_image(const std::string& name, const Image& img) {
        ensure(name);
        extract_vision(img, 0.4f, 0.4f, 0.45f, positions[name]);
    }

    // ═══════════════════════════════════════
    //  PHYSICS: interact with object
    // ═══════════════════════════════════════

    void phys_lift(const std::string& name, const PhysObj& obj, std::mt19937& rng) {
        ensure(name);
        std::normal_distribution<float> n(0, 0.08f);
        positions[name].d[P_WEIGHT]   = std::clamp(obj.density * obj.wall_mm * 0.01f + n(rng), 0.f, 1.f);
        positions[name].d[P_TEMP]     = std::clamp((obj.temperature - 20) / 15.f + n(rng) * 0.3f, -1.f, 1.f);
        positions[name].d[P_FRICTION] = std::clamp(obj.friction + n(rng) * 0.1f, 0.f, 1.f);
        positions[name].d[P_TEXTURE]  = std::clamp(obj.roughness + n(rng) * 0.2f, 0.f, 1.f);
    }

    void phys_tap(const std::string& name, const PhysObj& obj, std::mt19937& rng) {
        ensure(name);
        std::normal_distribution<float> n(0, 0.08f);
        positions[name].d[P_TAP_SOUND]   = std::clamp(obj.sound_hz / 5000.f + n(rng), 0.f, 1.f);
        positions[name].d[P_TAP_REBOUND] = std::clamp(obj.cor + n(rng) * 0.1f, 0.f, 1.f);
    }

    void phys_squeeze(const std::string& name, const PhysObj& obj, std::mt19937& rng) {
        ensure(name);
        std::normal_distribution<float> n(0, 0.08f);
        float resist = std::clamp(obj.youngs_gpa / 100 * (obj.wall_mm / 5) + n(rng), 0.f, 1.f);
        float deform = std::clamp(1.f / (obj.youngs_gpa + 1) * 5 + n(rng), 0.f, 1.f);
        float brit_r = obj.youngs_gpa / (obj.ult_mpa + 1);
        float thick_p = std::min(obj.wall_mm / 10.f, 1.f);
        float break_r = (1 - brit_r * 0.5f) * (0.3f + 0.7f * thick_p);
        bool broke = 0.8f > break_r && break_r < 0.5f;
        float spring = broke ? 0 : std::clamp(obj.elastic_limit * 10 + n(rng), 0.f, 1.f);

        positions[name].d[P_SQ_RESIST] = resist;
        positions[name].d[P_SQ_DEFORM] = deform;
        positions[name].d[P_SQ_SPRING] = spring;
        positions[name].d[P_SQ_BROKE]  = broke ? 1.f : 0.f;
        positions[name].d[D_BRITTLE]   = broke ? resist * (1 - break_r) : 0;
        positions[name].d[D_TOUGH]     = !broke ? resist * (1 - deform) : 0;
        positions[name].d[D_SOFT]      = (1 - resist) * deform * spring;
        positions[name].d[D_SLIPPERY]  = (1 - positions[name].d[P_FRICTION]) * 0.7f;
        if (positions[name].d[V_SIZE] > 0.01f)
            positions[name].d[D_DENSITY] = positions[name].d[P_WEIGHT] / (positions[name].d[V_SIZE] + 0.01f);
    }

    void phys_push(const std::string& name, const PhysObj& obj, std::mt19937& rng) {
        ensure(name);
        std::normal_distribution<float> n(0, 0.08f);
        positions[name].d[P_PUSH_FRICTION] = std::clamp(obj.friction * obj.density * 0.3f + n(rng), 0.f, 1.f);
        positions[name].d[P_SLIDE_SOUND]   = std::clamp(obj.friction * (1 - obj.roughness) + n(rng) * 0.1f, 0.f, 1.f);
    }

    // Full perceive: look at it, lift it, tap it, squeeze it, push it
    void perceive_full(const std::string& name, const PhysObj& obj, std::mt19937& rng) {
        auto img = obj.render(rng);
        see_image(name, img);
        phys_lift(name, obj, rng);
        phys_tap(name, obj, rng);
        phys_squeeze(name, obj, rng);
        phys_push(name, obj, rng);
    }

    // ═══════════════════════════════════════
    //  PHYSICS: experience outcome
    // ═══════════════════════════════════════

    void phys_experience(const std::string& action, const std::string& target, bool success) {
        time_step++;
        ensure(target);
        float val = success ? 1.f : -1.f;
        trajectories.push_back({action, target, val, time_step});
        tried[action].insert(target);

        int odim = -1;
        if (action == "drop")         odim = O_DROP;
        else if (action == "throw")   odim = O_THROW;
        else if (action == "squeeze") odim = O_SQUEEZE;
        else if (action == "push")    odim = O_PUSH;

        if (odim >= 0) {
            float cnt = positions[target].d[M_EXPERIENCE];
            positions[target].d[odim] = (positions[target].d[odim] * cnt + val) / (cnt + 1);
        }
        positions[target].d[M_EXPERIENCE] += 1;
        positions[target].d[M_FAMILIARITY] += 0.05f;
    }

    // ═══════════════════════════════════════
    //  PHYSICS: predict outcome
    // ═══════════════════════════════════════

    struct PhysPred { bool success; float confidence; float valence; int matches; };

    PhysPred phys_predict(const std::string& action, const std::string& target) {
        ensure(target);
        QVec& tp = positions[target];

        // If we've tried this exact action on this exact object, use direct memory
        if (tried[action].count(target)) {
            int odim = -1;
            if (action == "drop")         odim = O_DROP;
            else if (action == "throw")   odim = O_THROW;
            else if (action == "squeeze") odim = O_SQUEEZE;
            else if (action == "push")    odim = O_PUSH;
            if (odim >= 0) {
                float v = tp.d[odim];
                return {v >= 0, std::min(fabsf(v), 1.f), v, 1};
            }
        }

        // Generalize: compare dims 0-26 (vision+physics+derived)
        float pv = 0, nv = 0; int m = 0;
        for (auto& tr : trajectories) {
            if (tr.action != action || !positions.count(tr.target)) continue;
            float sim = overlap_ch(tp, positions[tr.target], 0, D_DENSITY);
            if (sim > 0.005f) {
                if (tr.outcome > 0) pv += sim; else nv += sim;
                m++;
            }
        }
        if (pv + nv < 0.001f) return {true, 0, 0, 0};
        float v = (pv - nv) / (pv + nv);
        return {v >= 0, std::min(fabsf(v), 1.f), v, m};
    }

    // ═══════════════════════════════════════
    //  CAUSAL: observe, predict, backward recall
    // ═══════════════════════════════════════

    void causal_observe(const std::string& cause, const std::string& effect, float strength = 1) {
        time_step++;
        ensure(cause);
        ensure(effect);
        causal_links.push_back({cause, effect, strength, time_step});

        QVec dir = positions[effect] - positions[cause];
        causal_fwd[cause].push_back({effect, dir});
        QVec rev = positions[cause] - positions[effect];
        causal_rev[effect].push_back({cause, rev});

        // Pull meaning dims gently
        QVec pull = (positions[effect] - positions[cause]) * 0.05f;
        positions[cause].d[T_MEANING_X] += pull.d[T_MEANING_X];
        positions[cause].d[T_MEANING_Y] += pull.d[T_MEANING_Y];

        // CONTEXT PROPAGATION: cause and effect SHARE context
        for (int cd = CTX_A; cd <= CTX_D; cd++) {
            float avg = (positions[cause].d[cd] + positions[effect].d[cd]) * 0.5f;
            positions[cause].d[cd] += (avg - positions[cause].d[cd]) * 0.2f;
            positions[effect].d[cd] += (avg - positions[effect].d[cd]) * 0.2f;
        }

        // Sibling tracking
        for (auto& link : causal_links) {
            if (link.cause == cause && link.effect != effect) {
                // Same cause, different effects: share context only
                for (int cd = CTX_A; cd <= CTX_D; cd++) {
                    float avg0 = (positions[effect].d[cd] + positions[link.effect].d[cd]) * 0.5f;
                    positions[effect].d[cd] += (avg0 - positions[effect].d[cd]) * 0.03f;
                }
            }
            if (link.effect == effect && link.cause != cause) {
                QVec d = positions[link.cause] - positions[cause];
                positions[cause].d[T_MEANING_X] += d.d[T_MEANING_X] * 0.03f;
                positions[cause].d[T_MEANING_Y] += d.d[T_MEANING_Y] * 0.03f;
                causal_siblings[cause].insert(link.cause);
                causal_siblings[link.cause].insert(cause);
                for (int cd = CTX_A; cd <= CTX_D; cd++) {
                    float avg2 = (positions[cause].d[cd] + positions[link.cause].d[cd]) * 0.5f;
                    positions[cause].d[cd] += (avg2 - positions[cause].d[cd]) * 0.1f;
                    positions[link.cause].d[cd] += (avg2 - positions[link.cause].d[cd]) * 0.1f;
                }
            }
        }
    }

    struct CausalPred {
        bool causes;
        float confidence;
        std::vector<std::string> chain;
        std::string method;
    };

    CausalPred causal_predict(const std::string& cause, const std::string& effect, int max_hops = 5) {
        ensure(cause);
        ensure(effect);
        if (max_hops <= 0) return {false, 0, {cause}, "depth"};

        float ctx_sim = overlap_ch(positions[cause], positions[effect], CTX_A, CTX_D);

        // Direct link
        for (auto& l : causal_links)
            if (l.cause == cause && l.effect == effect)
                return {l.strength > 0, fabsf(l.strength), {cause, effect}, "direct"};

        // Forward chain — only follow hops in similar context
        std::string cur = cause;
        std::vector<std::string> chain = {cause};
        std::unordered_set<std::string> visited = {cause};
        float cs = 1;

        for (int hop = 0; hop < max_hops; hop++) {
            if (!causal_fwd.count(cur) || causal_fwd[cur].empty()) break;
            std::string best_n; float bp = -999;

            for (auto& [next, dir] : causal_fwd[cur]) {
                if (visited.count(next)) continue;
                float hop_ctx = overlap_ch(positions[cause], positions[next], CTX_A, CTX_D);
                QVec pred = positions[cur] + dir;
                float db = sqrtf(positions[cur].dist_sq(positions[effect]));
                float da = sqrtf(pred.dist_sq(positions[effect]));
                float prog = db - da;
                float dov = overlap(positions[next], positions[effect]);
                float ctx_weight = 0.3f + 0.7f * hop_ctx;
                float score = (prog * 0.5f + dov * 0.5f) * ctx_weight;
                if (score > bp) { bp = score; best_n = next; }
            }

            if (best_n.empty()) break;
            visited.insert(best_n);
            chain.push_back(best_n);
            cs *= 0.8f;

            if (best_n == effect)
                return {true, cs, chain, "chain(" + std::to_string(hop + 1) + ")"};

            float ov = overlap(positions[best_n], positions[effect]);
            float near_ctx = overlap_ch(positions[best_n], positions[effect], CTX_A, CTX_D);
            if (ov > 0.5f && near_ctx > 0.2f) {
                chain.push_back("~" + effect);
                return {true, cs * ov, chain, "near(" + std::to_string(hop + 1) + ")"};
            }
            cur = best_n;
        }

        // Backward recall with sibling check
        if (causal_rev.count(effect)) {
            float bs = 0; std::string bkc;
            for (auto& [kc, _] : causal_rev[effect]) {
                if (kc == cause) continue;
                bool sib = causal_siblings.count(cause) && causal_siblings[cause].count(kc);
                float sim = overlap(positions[cause], positions[kc]);
                float bk_ctx = overlap_ch(positions[cause], positions[kc], CTX_A, CTX_D);
                if (sib) sim = std::min(sim + 0.3f, 1.f);
                else if (sim < 0.6f || bk_ctx < 0.25f) sim = 0;
                if (sim > bs) { bs = sim; bkc = kc; }
            }
            if (bs > 0.15f) {
                float ks = 0;
                for (auto& l : causal_links)
                    if (l.cause == bkc && l.effect == effect) ks = l.strength;
                chain.push_back("~" + bkc);
                chain.push_back(effect);
                return {ks > 0, bs * fabsf(ks) * 0.6f, chain, "backward(~" + bkc + ")"};
            }
            for (auto& [kc, _] : causal_rev[effect]) {
                float bc_ctx = overlap_ch(positions[cause], positions[kc], CTX_A, CTX_D);
                if (bc_ctx < 0.1f) continue;
                auto sub = causal_predict(cause, kc, max_hops - 2);
                if (sub.causes && sub.confidence > 0.1f) {
                    auto cc = sub.chain;
                    cc.push_back(effect);
                    return {true, sub.confidence * 0.7f, cc, "backward_chain(via " + kc + ")"};
                }
            }
        }

        return {false, 0, chain, "no_path"};
    }

    CausalPred causal_predict_ctx(const std::string& cause, const std::string& effect,
            const std::vector<std::string>& ctx) {
        for (auto& c : ctx) {
            for (auto& l : causal_links)
                if (l.cause == c && l.effect == effect && l.strength < 0)
                    return {false, fabsf(l.strength), {cause, c + "(blocks)", effect}, "blocked"};
            for (auto& l2 : causal_links)
                if (l2.cause == cause && l2.effect != effect)
                    for (auto& l3 : causal_links)
                        if (l3.cause == c && l3.effect == l2.effect && l3.strength < 0)
                            return {false, fabsf(l3.strength) * 0.7f,
                                    {cause, l2.effect + "(blocked)", effect}, "int_blocked"};
        }
        return causal_predict(cause, effect);
    }

    // ═══════════════════════════════════════
    //  AUTONOMOUS HEARTBEAT
    // ═══════════════════════════════════════

    void give_input(const std::string& text) {
        std::lock_guard<std::mutex> lock(input_mutex);
        pending_inputs.push_back(split(text));
    }

    struct HBResult { int ticks, actions; std::vector<std::string> log; };

    HBResult heartbeat(int n = 10, bool verbose = false) {
        HBResult res; res.ticks = n; res.actions = 0;

        for (int tick = 0; tick < n; tick++) {
            auto_tick++; time_step++;
            std::vector<std::string> inp;

            {
                std::lock_guard<std::mutex> lock(input_mutex);
                if (!pending_inputs.empty()) {
                    inp = pending_inputs.front();
                    pending_inputs.erase(pending_inputs.begin());
                }
            }

            if (!inp.empty()) {
                auto th = think(inp, false);
                std::string msg = "input: " + th.answer;
                res.log.push_back(msg);
                res.actions++;
                if (verbose) printf("  [%d] %s\n", auto_tick, msg.c_str());
                energy -= 0.05f;
            } else {
                auto [pk, pe] = field_peak(meta_field);
                if (pe > 0.5f && auto_tick - last_resolve >= 5 && energy > 0.1f) {
                    self_repair();
                    for (auto& d : meta_field) d.weight *= 0.5f;
                    last_resolve = auto_tick;
                    res.log.push_back("resolve");
                    res.actions++;
                    energy -= 0.08f;
                } else if (boredom > 0.3f && energy > 0.1f) {
                    std::vector<std::string> cw;
                    for (auto& [w, p] : positions)
                        if (p.d[T_WEIGHT] > 0.3f && !is_stop(w)) cw.push_back(w);
                    if (!cw.empty()) {
                        std::mt19937 rng(time_step);
                        auto& w = cw[rng() % cw.size()];
                        auto ch = reason({w}, 3);
                        float q = evaluate(ch.words);
                        res.log.push_back("explore: " + w);
                        res.actions++;
                    }
                    boredom = std::max(0.f, boredom - 0.3f);
                    energy -= 0.03f;
                } else {
                    boredom += 0.1f;
                    energy = std::min(1.f, energy + 0.05f);
                }
            }
        }
        return res;
    }

    void live(float tick_interval = 0.1f) {
        alive = true;
        while (alive) {
            heartbeat(1, false);
            std::this_thread::sleep_for(
                std::chrono::milliseconds((int)(tick_interval * 1000)));
        }
    }

    void stop() { alive = false; }

    // ═══════════════════════════════════════
    //  SAVE / LOAD
    // ═══════════════════════════════════════

    void save_state(const std::string& path) {
        std::ofstream f(path);
        f << training_data.size() << "\n";
        for (auto& [name, tokens] : training_data) {
            f << name << "\t";
            for (int i = 0; i < (int)tokens.size(); i++)
                f << (i ? " " : "") << tokens[i];
            f << "\n";
        }
        f << time_step << "\n";
    }

    void load_state(const std::string& path) {
        std::ifstream f(path);
        int n; f >> n; f.ignore();
        training_data.clear();
        for (int i = 0; i < n; i++) {
            std::string line;
            std::getline(f, line);
            auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            training_data.push_back({line.substr(0, tab), split(line.substr(tab + 1))});
        }
        f >> time_step;
        train(training_data);
    }

    // ═══════════════════════════════════════
    //  SUMMARY
    // ═══════════════════════════════════════

    void summary() {
        printf("=== UNIFIED QUANTUM FIELD AI ===\n");
        printf("  Dims: %d | Entities: %d | Concepts: %d\n",
               QDIMS, (int)positions.size(), (int)concept_names.size());
        printf("  Memories: %d | Trajectories: %d | Causal: %d\n",
               (int)memory_field.size(), (int)trajectories.size(), (int)causal_links.size());
        printf("  Time: %d\n", time_step);
    }
};

// ═══════════════════════════════════════════════════════════════
//  MAIN — Example usage
// ═══════════════════════════════════════════════════════════════

int main() {
    UnifiedAI ai;

    // --- Train text classification ---
    ai.train({
        {"sports",  split("the team scored a goal in the final minute")},
        {"sports",  split("the player kicked the ball into the net")},
        {"sports",  split("the coach led the team to victory")},
        {"science", split("the researchers discovered a new species")},
        {"science", split("the experiment confirmed the hypothesis")},
        {"science", split("the telescope observed a distant galaxy")},
    });

    // --- Classify ---
    auto result = ai.classify(split("the team scored a goal"));
    printf("Classification: %s (%.1f%% confidence)\n",
           result.best.c_str(), result.confidence * 100);

    // --- Causal reasoning ---
    for (int r = 0; r < 5; r++) {
        ai.causal_observe("rain", "wet");
        ai.causal_observe("wet", "slip");
        ai.causal_observe("slip", "fall");
    }
    auto pred = ai.causal_predict("rain", "fall");
    printf("Does rain cause fall? %s (%.2f confidence, method: %s)\n",
           pred.causes ? "YES" : "NO", pred.confidence, pred.method.c_str());

    // --- Chat ---
    printf("Chat: %s\n", ai.chat("the team scored a goal").c_str());
    printf("Chat: %s\n", ai.chat("the researchers discovered something").c_str());

    // --- Introspect ---
    printf("%s", ai.introspect().c_str());

    return 0;
}

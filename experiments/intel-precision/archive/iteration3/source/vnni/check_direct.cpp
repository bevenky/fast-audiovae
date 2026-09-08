// Focused arithmetic and dispatch checker, without timing, ORT, models or GPUs.
#include "direct_vnni.h"
#include <array>
#include <future>
#include <iostream>
#include <random>
#include <vector>

using ip3_direct_vnni::Arguments;
using ip3_direct_vnni::check;

static bool bits(float a, float b) {
    uint32_t aa, bb;
    std::memcpy(&aa, &a, sizeof(aa)); std::memcpy(&bb, &b, sizeof(bb));
    return aa == bb;
}

static void same(const std::vector<float>& a, const std::vector<float>& b) {
    check(a.size() == b.size(), "Shape mismatch");
    for (size_t i = 0; i < a.size(); ++i) check(bits(a[i], b[i]), "FP32 bit mismatch");
}

static std::vector<float> reference(const Arguments& a, const std::vector<int8_t>& x) {
    std::vector<float> out(size_t(a.m) * a.t, 12345.f);
    for (int m = a.first; m < a.last; ++m) for (int t = 0; t < a.t; ++t) {
        int64_t corrected = 0;
        for (int k = 0; k < a.k; ++k)
            corrected += (int(a.weights[size_t(m) * a.k + k]) - 128) * int(x[size_t(k) * a.t + t]);
        check(corrected >= INT32_MIN && corrected <= INT32_MAX, "Reference overflow");
        const float scale = a.sw[m] * a.sx[t];
        float value = float(static_cast<int32_t>(corrected)) * scale;
        if (a.bias) { const float biased = value + a.bias[m]; value = a.skip[size_t(m) * a.t + t] + biased; }
        out[size_t(m) * a.t + t] = value;
    }
    return out;
}

static int test_shape(int m, int k, int t, int seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> w(size_t(m) * k);
    std::vector<int8_t> x(size_t(k) * t), packed(x.size(), 42);
    std::vector<float> sw(m), sx(t), bias(m), skip(size_t(m) * t), out(skip.size());
    std::vector<int32_t> sums(t, 0);
    for (auto& value : w) value = static_cast<uint8_t>(rng() % 256);
    for (int row = 0; row < k; ++row) for (int time = 0; time < t; ++time) {
        const int value = int(rng() % 255) - 127;
        x[size_t(row) * t + time] = static_cast<int8_t>(value); sums[time] += value;
    }
    for (auto& value : sw) value = float(1 + rng() % 47) / 4096.f;
    for (auto& value : sx) value = float(1 + rng() % 37) / 8192.f;
    for (auto& value : bias) value = float(int(rng() % 2001) - 1000) / 8192.f;
    for (auto& value : skip) value = float(int(rng() % 2001) - 1000) / 4096.f;
    const auto old_x = x; const auto old_w = w;
    check(ip3_direct_vnni::pack(x.data(), x.size(), k, t, packed.data(), packed.size(), 1, 3), "Pack rejected supported shape");
    for (int row = 0; row < k; ++row) for (int time = 0; time < t; ++time)
        check(packed[(size_t(row / 4) * t + time) * 4 + row % 4] == x[size_t(row) * t + time], "Incorrect packed byte order");
    int cases = 0;
    for (bool residual : {false, true}) {
        Arguments a{m, k, t, 0, m, w.data(), packed.data(), sw.data(), sx.data(), sums.data(), out.data(),
                    residual ? bias.data() : nullptr, residual ? skip.data() : nullptr};
        const auto wanted = reference(a, x);
        for (int rows : {4, 8}) {
            std::fill(out.begin(), out.end(), 12345.f);
            check(ip3_direct_vnni::run(a, 1, 3, rows), "Run rejected supported shape");
            same(out, wanted); ++cases;
            if (m >= 12) {
                a.first = 4; a.last = m-4;
                std::fill(out.begin(), out.end(), 12345.f);
                check(ip3_direct_vnni::run(a, 1, 3, rows), "Partial row run rejected");
                same(out, reference(a, x)); ++cases;
                a.first = 0; a.last = m;
            }
        }
    }
    check(old_x == x && old_w == w, "Input bytes changed");
    return cases;
}

int main() {
    try {
        if (!ip3_direct_vnni::host_available()) {
            std::cout << "{\"status\":\"skipped\",\"reason\":\"AVX512 VNNI CPU and OS support unavailable\",\"timings_collected\":false}\n";
            return 0;
        }
        const std::array<std::array<int,3>,17> shapes{{
            {4,4,16},{4,8,32},{8,12,16},{12,4,64},{12,12,32},{32,32,16},
            {64,64,64},{128,128,128},{256,256,256},{256,128,128},{128,256,64},
            {4,256,256},{256,4,16},{32,128,256},{32,32,48},{64,64,96},{128,128,240}}};
        int arithmetic = 0;
        for (size_t i = 0; i < shapes.size(); ++i)
            arithmetic += test_shape(shapes[i][0], shapes[i][1], shapes[i][2], int(917+i));

        std::vector<uint8_t> w(4*4, 128);
        std::vector<int8_t> x(4*16, 1), packed(x.size());
        std::vector<float> sw(4, float(1u<<26)), sx(16, 1.f), bias(4, 1.f), skip(4*16, -float(1u<<26)), out(4*16);
        std::vector<int32_t> sums(16, 4);
        for (int row = 0; row < 4; ++row) w[row*4] = 129;
        check(ip3_direct_vnni::pack(x.data(), x.size(), 4, 16, packed.data(), packed.size(), 1, 3), "Fixture packing failed");
        Arguments a{4,4,16,0,4,w.data(),packed.data(),sw.data(),sx.data(),sums.data(),out.data(),bias.data(),skip.data()};
        for (int rows : {4, 8}) {
            check(ip3_direct_vnni::run(a,1,3,rows), "Cancellation fixture rejected");
            same(out, reference(a,x));
            for (float value : out) check(bits(value, 0.f), "Bias/skip addition reassociated");
        }
        std::fill(sw.begin(),sw.end(),1e-30f); std::fill(sx.begin(),sx.end(),1e-30f);
        std::fill(bias.begin(),bias.end(),-0.f); std::fill(skip.begin(),skip.end(),-0.f);
        for (int row = 0; row < 4; ++row) w[row*4] = row % 2 ? 129 : 127;
        check(ip3_direct_vnni::run(a,1,3), "Signed-zero fixture rejected"); same(out,reference(a,x));
        check(std::signbit(out[0]) && !std::signbit(out[16]), "Signed-zero order lost");

        int fallback = 0;
        for (auto dims : std::array<std::array<int,3>,7>{{{1,4,16},{4,3,16},{4,4,15},{260,4,16},{4,260,16},{4,4,272},{4,4,0}}}) {
            auto b = a; b.m=dims[0]; b.k=dims[1]; b.t=dims[2]; b.last=b.m;
            std::fill(out.begin(),out.end(),12345.f);
            check(!ip3_direct_vnni::run(b,1,3), "Unsupported shape dispatched");
            for (float value:out) check(value==12345.f,"Fallback touched output"); ++fallback;
        }
        for (auto interval:std::array<std::array<int,2>,3>{{{1,4},{0,3},{4,0}}}) {
            auto b=a;b.first=interval[0];b.last=interval[1];
            check(!ip3_direct_vnni::run(b,1,3),"Unsupported row interval dispatched");++fallback;
        }
        check(!ip3_direct_vnni::run(a,0,3),"Scalar backend forced VNNI");++fallback;
        check(!ip3_direct_vnni::run(a,1,0),"Unavailable capability forced VNNI");++fallback;
        check(!ip3_direct_vnni::run(a,1,3,3),"Unsupported register block dispatched");++fallback;
        check(!ip3_direct_vnni::pack(x.data(),x.size(),4,16,packed.data(),packed.size(),1,0),"Unavailable capability packed");++fallback;
        int rejected = 0;
        try { ip3_direct_vnni::pack(x.data(),x.size(),4,16,x.data(),x.size(),1,3); }
        catch(const std::invalid_argument&) { ++rejected; }
        try { ip3_direct_vnni::pack(x.data(),x.size(),4,16,packed.data(),packed.size()-1,1,3); }
        catch(const std::invalid_argument&) { ++rejected; }
        auto bad=a;bad.skip=nullptr;
        try { ip3_direct_vnni::run(bad,1,3); } catch(const std::invalid_argument&) { ++rejected; }
        bad=a;bad.output=const_cast<float*>(bad.skip);
        try { ip3_direct_vnni::run(bad,1,3); } catch(const std::invalid_argument&) { ++rejected; }
        bias[0]=std::numeric_limits<float>::infinity();
        try { ip3_direct_vnni::run(a,1,3); } catch(const std::invalid_argument&) { ++rejected; }
        check(rejected==5,"Malformed inputs were not rejected");

        auto first=std::async(std::launch::async,[]{return test_shape(128,128,128,333);});
        auto second=std::async(std::launch::async,[]{return test_shape(128,128,64,334);});
        const int concurrent=first.get()+second.get();
        std::cout << "{\"status\":\"passed\",\"gpu_used\":false,\"timings_collected\":false,\"models_executed\":false,"
                  << "\"shape_cases\":" << shapes.size() << ",\"arithmetic_checks\":" << arithmetic
                  << ",\"concurrent_arithmetic_checks\":" << concurrent << ",\"fallback_checks\":" << fallback
                  << ",\"malformed_checks\":" << rejected << ",\"cancellation_and_signed_zero\":true}\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "Direct VNNI check failed: " << error.what() << '\n'; return 1;
    }
}

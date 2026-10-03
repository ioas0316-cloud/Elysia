#include <elysia/elysia_holographic_engine.hpp>
#include <iostream>
#include <cassert>
#include <fstream>
#include <cstdio>

void test_morton_encoding() {
    std::cout << "[Test 1/3] Testing 3D Morton Z-Order Encoding and Decoding..." << std::endl;

    uint32_t x_in = 15, y_in = 27, z_in = 42;
    uint64_t code = elysia::encodeMorton3D(x_in, y_in, z_in);

    uint32_t x_out = 0, y_out = 0, z_out = 0;
    elysia::decodeMorton3D(code, x_out, y_out, z_out);

    assert(x_in == x_out && "Morton X decoding failed!");
    assert(y_in == y_out && "Morton Y decoding failed!");
    assert(z_in == z_out && "Morton Z decoding failed!");

    std::cout << "  Passed: Morton (15, 27, 42) -> Code 0x" << std::hex << code
              << " -> Decoded (" << std::dec << x_out << ", " << y_out << ", " << z_out << ")" << std::endl;
}

void test_hangul_kronecker() {
    std::cout << "[Test 2/3] Testing Hangul Jamo Kronecker Tensor Encoding & Topological Distance..." << std::endl;

    uint32_t char1 = 0xD55C; // '한'
    uint32_t char2 = 0xD560; // '할'

    uint64_t t1 = elysia::encodeHangulKronecker(char1);
    uint64_t t2 = elysia::encodeHangulKronecker(char2);

    assert(t1 != 0ULL && "'한' Kronecker tensor should not be zero");
    assert(t2 != 0ULL && "'할' Kronecker tensor should not be zero");

    int dist = elysia::computeTopologicalBitDistance(t1, t2);
    int self_dist = elysia::computeTopologicalBitDistance(t1, t1);

    assert(self_dist == 0 && "Self-distance must be 0");
    assert(dist >= 0 && dist <= 64 && "Bit distance must be in [0, 64]");

    std::cout << "  Passed: '한' Tensor = 0x" << std::hex << t1
              << ", '할' Tensor = 0x" << t2
              << ", Bit Distance = " << std::dec << dist << " bits" << std::endl;
}

void test_zero_copy_header_and_buffer() {
    std::cout << "[Test 3/3] Testing 128-byte TopologicalHeader and ZeroCopyTopologicalBuffer..." << std::endl;

    assert(sizeof(elysia::TopologicalHeader) == 128 && "Header size must be exactly 128 bytes");

    const char* test_file = "test_topo_field.elys";
    {
        std::ofstream ofs(test_file, std::ios::binary);
        elysia::TopologicalHeader header;
        std::memset(&header, 0, sizeof(header));
        header.magic_number = 0x454C5953; // 'ELYS'
        header.dimension = 3;
        header.grid_dim[0] = 128;
        header.grid_dim[1] = 128;
        header.grid_dim[2] = 128;
        header.spatial_step_dx = 0.5f;
        header.payload_bytes = 1024 * sizeof(float);

        ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));
        std::vector<float> dummy_data(1024, 1.0f);
        ofs.write(reinterpret_cast<const char*>(dummy_data.data()), dummy_data.size() * sizeof(float));
    }

    auto* buf = elysia::ZeroCopyTopologicalBuffer::MapFromFile(test_file);
    assert(buf != nullptr && "mmap from file failed");
    assert(buf->header->magic_number == 0x454C5953 && "Magic number mismatch");
    assert(buf->header->grid_dim[0] == 128 && "Grid dimension X mismatch");
    assert(buf->tensor_data != nullptr && "Tensor data pointer is null");

    delete buf;
    std::remove(test_file);

    std::cout << "  Passed: ZeroCopyTopologicalBuffer successfully mapped and verified." << std::endl;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "Running Elysia Holographic Engine C++ Unit Tests" << std::endl;
    std::cout << "==========================================================" << std::endl;

    test_morton_encoding();
    test_hangul_kronecker();
    test_zero_copy_header_and_buffer();

    std::cout << "==========================================================" << std::endl;
    std::cout << "ALL HOLOGRAPHIC ENGINE TESTS PASSED!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}

#pragma once
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

uint32_t swap32(uint32_t v) {
    return (v >> 24) |
        ((v >> 8) & 0x0000FF00) |
        ((v << 8) & 0x00FF0000) |
        (v << 24);
}

uint32_t readBigEndian(ifstream& file) {
    uint32_t result = 0;
    file.read(reinterpret_cast<char*>(&result),sizeof(result));
    return swap32(result);
}

void read_dataset(vector<vector<double>>& Y) {
    ifstream file("..\\train-images.idx3-ubyte",ios::binary);
    if(!file) {
        cerr << "Cannot open image file\n";
    }

    uint32_t magic = readBigEndian(file);
    uint32_t numImages = readBigEndian(file);
    uint32_t rows = readBigEndian(file);
    uint32_t cols = readBigEndian(file);

    vector<uint8_t> images(numImages * rows * cols);
    file.read(reinterpret_cast<char*>(images.data()),images.size());

    cout << "Magic: " << magic << "\n";
    cout << "Images: " << numImages << "\n";
    cout << "Size: " << rows << "x" << cols << "\n";


    Y.clear();

    Y.resize(numImages);
    for(int i = 0; i < numImages; i++) {
        for(int j = 0; j < (28 * 28); j++) {
            uint8_t pixel = images[
                i * 28 * 28 + j
            ];
            Y[i].push_back(pixel / 255.0);
        }
    }
}

void read_dataset_labels(vector<vector<double>>& X,vector<uint8_t>& labels) {
    ifstream file("..\\train-labels.idx1-ubyte",ios::binary);

    uint32_t magic = readBigEndian(file);
    uint32_t numLabels = readBigEndian(file);

    labels = vector<uint8_t>(numLabels);
    file.read(reinterpret_cast<char*>(labels.data()),labels.size());

    X.resize(numLabels);

    for(int i = 0; i < numLabels; i++) {
        uint8_t label = labels[i];
        for(int j = 0; j < 10; j++) {
            X[i].push_back(j == label);
        }
    }
}
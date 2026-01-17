#define SDL_MAIN_HANDLED
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <numeric>
#include <SDL2/SDL.h>
#include "mnist-reader.h"

const int W = 840,H = 840;

using namespace std;

typedef double(*x_func)(double);

static mt19937 global_rng(random_device{}());

double relu(double x) {
	return max(x,0.0);
}

double d_relu(double x) {
	return x > 0 ? 1 : 0;
}

double linear(double x) {
	return x;
}

double d_linear(double x) {
	return 1;
}

double d_tanh(double x) {
	return (1 - pow(tanh(x),2));
}

double msq_error(double a,double b) {
	return (a - b) * (a - b);
}

vector<double> softmax(const vector<double>& z) {
	vector<double> result(z.size());
	double max_val = *max_element(z.begin(),z.end()); // for stability
	double sum = 0.0;

	for(size_t i = 0; i < z.size(); i++) {
		result[i] = exp(z[i] - max_val);
		sum += result[i];
	}

	for(size_t i = 0; i < z.size(); i++) {
		result[i] /= sum;
	}

	return result;
}

class Neuron {
public:
	vector<double> weights;
	double bias = 0;

	// training temp values
	double sum_bias = 0;
	vector<double> sum_weights;

	double dev_cost_act = 0;

	Neuron() {};
	Neuron(size_t previous_layer_size) {
		bias = 0;
		dev_cost_act = 0;
		weights.resize(previous_layer_size);

		double stddev = sqrt(2.0 / previous_layer_size); // He initialization
		normal_distribution<double> distribution(0.0, stddev);

		for(size_t i = 0; i < previous_layer_size; i++) {
			weights[i] = distribution(global_rng);
		}
		bias = 0;
	}
	void evaluate(const vector<double>& values,double(*activ_f)(double),double& activation,double& linear) {
		if(values.size() != weights.size()) throw "previous layer values and weights size doesnt match";
		linear = bias;
		for(size_t i = 0; i < values.size(); i++) {
			linear += weights[i] * values[i];
		}
		activation = activ_f(linear);
	}
	void init_wb() {
		sum_weights.assign(weights.size(),0.0);
		sum_bias = 0.0;
	}
	void apply_wb(double lr,int sample_size) {
		for(int i = 0; i < sum_weights.size(); i++) {
			weights[i] -= (sum_weights[i] / sample_size) * lr;
		}
		bias -= (sum_bias / sample_size) * lr;
	}
	void draw_weight(SDL_Renderer* renderer) {
		SDL_SetRenderDrawColor(renderer,0,0,0,255);
		SDL_RenderClear(renderer);
		double side = sqrt(weights.size());
		int _x = floor(side),_y = weights.size() / _x;

		int quad_x = W / _x;
		int quad_y = H / _y;

		double w_max = max(*max_element(weights.begin(),weights.end()),abs(*min_element(weights.begin(),weights.end())));

		int w_idx = 0;
		for(int i = 0; i < W; i+=quad_x) {
			for(int j = 0; j < H; j += quad_y) {
				SDL_SetRenderDrawColor(renderer,0,0,0,255);
				SDL_Rect r{i,j,quad_x,quad_y};

				double w = weights[w_idx];

				if(w < 0) {
					SDL_SetRenderDrawColor(renderer,(abs(w) / w_max) * 255,0,0,255); // red
				}
				else if(w > 0) {
					SDL_SetRenderDrawColor(renderer,0,(w / w_max) * 255,0,255); // green
				}

				SDL_RenderFillRect(renderer,&r);

				w_idx++;
				if(w_idx >= weights.size()) goto end;
			}
		}
	end:
		SDL_RenderPresent(renderer);
	}
};

class Layer {
public:
	vector<double> activations,linear;
	vector<Neuron> neurons;
	x_func activation; x_func d_activation;
	Layer() {};
	Layer(size_t size,size_t previous_size,x_func _activation,x_func _d_activation) {
		activation = _activation;
		d_activation = _d_activation;
		neurons.resize(size);
		for(int i = 0; i < size; i++) {
			neurons[i] = (Neuron(previous_size));
		}
		activations.resize(size);
		linear.resize(size);
	}
	void evaluate(vector<double> values) {
		for(size_t i = 0; i < neurons.size(); i++) {
			neurons[i].evaluate(values,activation,activations[i],linear[i]);
		}
	}
};

class Net {
public:
	int input_size;
	vector<Layer> layers; // hidden layers
	Net(const vector<int>& size,const vector<x_func>& activations,vector<x_func> d_activations) {
		if(activations.size() != (size.size() - 1)) throw "activations and net size dont match";
		input_size = size.front();
		layers.clear();
		layers.resize(size.size() - 1);
		for(size_t i = 1; i < size.size(); i++) {
			layers[i - 1] = (Layer(size[i],size[i - 1],activations[i - 1],d_activations[i - 1]));
		}
	}
	inline size_t output_size() const {
		return layers.back().neurons.size();
	}
	void feed_forward(vector<double>& values,bool use_soft_max = true) {
		for(size_t i = 0; i < layers.size(); i++) {
			layers[i].evaluate(values);
			if(i == (layers.size() - 1) && use_soft_max) {
				layers[i].activations = softmax(layers[i].activations);
			}
			values = layers[i].activations;
		}
	}
	double accuracy(const vector<vector<double>>& X,const vector<uint8_t>& Y_labels,size_t test_size = 100) {
		int correct = 0;
		for(int i = 0; i < min(test_size,X.size()); i++) {
			vector<double> result = X[i];
			feed_forward(result);
			auto iter = max_element(result.begin(),result.end());
			int result_n = distance(result.begin(),iter);
			if((uint8_t)result_n == Y_labels[i]) correct++;
		}
		return double(correct) / min(test_size,X.size());
	}
	void back_prop(vector<vector<double>>& X,vector<vector<double>>& Y,int batch_size,double lr) {
		if(Y[0].size() != output_size()) throw "Y size and Net ouput doesnt match";
		if(X[0].size() != input_size) throw "X size and Net input doesnt match";
		if(X.size() != Y.size()) throw "X and Y size doesnt match";

		vector<int> shuffled_indecies; shuffled_indecies.resize(X.size());

		iota(shuffled_indecies.begin(),shuffled_indecies.end(),0);

		random_device rd; // random generator for shuffling the indecies
		mt19937 gen(rd());

		shuffle(shuffled_indecies.begin(),shuffled_indecies.end(),gen);


		for(int b = 0; b < X.size(); b+=batch_size) { // look each batch

			for(Layer& _L : layers) {
				for(Neuron& L_N : _L.neurons) {
					L_N.init_wb();

				}
			}


			int end = min((int)X.size(),b + batch_size);

			for(int i = b; i < end; i++) {
				int t_i = shuffled_indecies[i];


				vector<double> net_output = X[t_i];
				feed_forward(net_output); // update network values according to current training sample

				for(int _l = 0; _l < layers.size(); _l++) { // loop each layer
					int l = layers.size() - _l - 1;
					Layer& layer = layers[l];

					vector<double> previous_layer = l > 0 ? layers[l - 1].activations : X[t_i];

					for(int ne = 0; ne < layer.neurons.size(); ne++) { // loop each neuron

						Neuron& neuron = layer.neurons[ne];

						if(_l == 0) { // calculate correctly the derivative of the cost over the neuron activation value
							neuron.dev_cost_act = (net_output[ne] - Y[t_i][ne]);
						}
						else {
							neuron.dev_cost_act = 0;
							for(int next_ne = 0; next_ne < layers[l + 1].neurons.size();next_ne++) {
								// the derivative of the cost based on the activation is calculated by making the sum of the chain rule for each neuron in the layer 
								neuron.dev_cost_act += layers[l+1].neurons[next_ne].dev_cost_act *
									layers[l + 1].d_activation(layers[l+1].linear[next_ne]) *
									layers[l + 1].neurons[next_ne].weights[ne]; // weight that multiplied the current iterating neuron (ne)
							}
						}

						double d_cost_bias = layer.d_activation(layer.linear[ne]) * neuron.dev_cost_act;
						neuron.sum_bias += d_cost_bias;
						for(int w = 0; w < neuron.weights.size(); w++) {
							neuron.sum_weights[w] += previous_layer[w] * d_cost_bias;
						}

					}

				}

			}

			for(Layer& _L : layers) {
				for(Neuron& L_N : _L.neurons) {
					L_N.apply_wb(lr,end-b);

				}
			}

		} 

	}
};

int main() {
	SDL_Init(SDL_INIT_VIDEO);
	SDL_Window* win; SDL_Renderer* ren;
	SDL_CreateWindowAndRenderer(W,H,0,&win,&ren);


	srand(time(0));
	Net rete({784,16,16,10},{relu,relu,linear},{d_relu,d_relu,d_linear});

	vector<vector<double>> X,Y;
	vector<uint8_t> Y_labels;
	
	read_dataset(X);
	read_dataset_labels(Y,Y_labels);

	int epochs = 8;
	for(int epoch = 0;epoch<epochs;epoch++) {
		printf("Epoch: %d, accuracy: %f\n",epoch,rete.accuracy(X,Y_labels,6000));
		//rete.layers[1].neurons[0].draw_weight(ren);
		rete.back_prop(X,Y,100,0.1);
		
	}


	vector<double> buffer;

	buffer.resize(28 * 28);

	for(double& v : buffer) {
		v = 0;
	}

	int quad_x = W / 28;
	int quad_y = H / 28;

	SDL_Event e;
	int mouse_x,mouse_y;

	while(1) {
		SDL_GetMouseState(&mouse_x,&mouse_y);

		mouse_x = SDL_clamp(mouse_x,10,W - 10),mouse_y=SDL_clamp(mouse_y,10,H-10);
		
		while(SDL_PollEvent(&e)) {

			double& c = buffer[(mouse_x / quad_x) + (mouse_y / quad_y) * 28],new_c;
			if(e.button.button == 1) {
				c = min(c + 0.01,1.0);
			}
			else if(e.button.button == 4) {
				c = max(c - 0.01,0.0);
			}

			

			if(e.type == SDL_KEYDOWN) {
				if(e.key.keysym.scancode == SDL_SCANCODE_C) {
					for(double& v : buffer) {
						v = 0;
					}
				}
			}
		}

		for(int i = 0; i < W; i+=quad_x) {
			for(int j = 0; j < H; j+=quad_y) {
				SDL_Rect r = {i,j,quad_x,quad_y};

				uint8_t col = buffer[i/quad_x + (j/quad_y) * 28] * 255;
				SDL_SetRenderDrawColor(ren,col,col,col,255);

				SDL_RenderFillRect(ren,&r);
			}
		}
		vector<double>result = buffer;
		rete.feed_forward(result);
		int result_n = distance(result.begin(),max_element(result.begin(),result.end()));

		cout << "Network output: " << result_n << endl;

		SDL_RenderPresent(ren);
	}


	return 0;
}
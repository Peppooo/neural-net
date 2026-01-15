#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>

using namespace std;

typedef double(*x_func)(double);

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

double random() {
	return 0.5*(rand() / (double)RAND_MAX)-0.25;
}

void randomize(vector<double>& v,size_t size) {
	v.clear();
	for(int i = 0; i < size; i++) {
		v.push_back(random());
	}
}

class Neuron {
public:
	double activation; // activation
	double linear; // non activated 
	vector<double> weights;
	double bias;
	
	// training temp values
	double sum_bias;
	vector<double> sum_weights;

	double dev_cost_act;


	Neuron(size_t previous_layer_size,bool _randomize = true) {
		bias = 0,activation = 0,linear = 0;
		if(_randomize) {
			randomize(weights,previous_layer_size);
			bias = random();
		}
	}
	void evaluate(const vector<double>& values,double(*activ_f)(double)) {
		if(values.size() != weights.size()) throw "previous layer values and weights size doesnt match";
		linear = bias;
		for(size_t i = 0; i < values.size(); i++) {
			linear += weights[i] * values[i];
		}
		activation = activ_f(linear);
	}
	void init_wb() {
		sum_weights.resize(weights.size());
		sum_bias = 0;
		for(int i = 0; i < weights.size(); i++) {
			sum_weights[i] = 0;
		}
	}
	void apply_wb(double lr,int sample_size) {
		for(int i = 0; i < sum_weights.size(); i++) {
			weights[i] -= (sum_weights[i] / sample_size)*lr;
		}
		bias -= (sum_bias / sample_size)*lr;
	}
};

class Layer {
public:
	vector<Neuron> neurons;
	x_func activation; x_func d_activation;
	Layer(size_t size,size_t previous_size,x_func _activation,x_func _d_activation) {
		activation = _activation;
		d_activation = _d_activation;
		neurons.clear();
		for(int i = 0; i < size;i++) {
			neurons.push_back(Neuron(previous_size));
		}
	}
	vector<double> output() {
		vector<double> out(neurons.size());
		for(size_t i = 0; i < neurons.size(); i++) {
			out[i] = neurons[i].activation;
		}
		return out;
	}
	void evaluate(vector<double> values) {
		for(size_t i = 0; i < neurons.size(); i++) {
			neurons[i].evaluate(values,activation);
		}
	}
};

class Net {
public:
	int input_size;
	vector<Layer> layers; // hidden layers
	Net(vector<int> size,vector<x_func> activations,vector<x_func> d_activations) {	
		if(activations.size() != (size.size() - 1)) throw "activations and net size dont match";
		input_size = size.front();
		layers.clear();
		for(size_t i = 1; i < size.size();i++) {
			layers.push_back(Layer(size[i],size[i-1],activations[i-1],d_activations[i-1]));
		}
	}
	inline size_t output_size() const {
		return layers.back().neurons.size();
	}
	vector<double> feed_forward(vector<double> values) {
		for(size_t i = 0; i < layers.size(); i++) {
			layers[i].evaluate(values);
			values = layers[i].output();
		}
		return values;
	}
	double net_error(vector<vector<double>> X,vector<vector<double>> Y) {
		if(Y[0].size() != output_size()) throw "Y size and Net ouput doesnt match";
		if(X[0].size() != Y[0].size() || X.size() != Y.size()) throw "Y size and X size doesnt match";

		double samples_sum = 0;
		for(int i = 0; i < X.size(); i++) {
			feed_forward(X[i]);
			vector<double> net_output = layers.back().output();
			double outputs_sum = 0;
			for(int j = 0; j < output_size(); j++) {
				outputs_sum += msq_error(net_output[j],Y[i][j]);
			}

			samples_sum += outputs_sum / X[0].size();
		}
		return samples_sum/X.size();
	}
	void back_prop(vector<vector<double>> X,vector<vector<double>> Y,double lr) {
		if(Y[0].size() != output_size()) throw "Y size and Net ouput doesnt match";
		if(X[0].size() != input_size) throw "X size and Net input doesnt match";
		if(X.size() != Y.size()) throw "X and Y size doesnt match";

		for(int i = 0; i < layers.size(); i++) {
			for(int j = 0; j < layers[i].neurons.size(); j++) {
				layers[i].neurons[j].init_wb();
			}
		}

		for(int i = 0; i < X.size(); i++) { // loop each training sample
			
			feed_forward(X[i]); // update network values according to current training sample

			for(int j = (int(layers.size()) - 1); j >= 0; j--) { // loop each layer

				Layer& layer = layers[j];

				vector<double> previous_layer = j > 0 ? layers[j - 1].output() : X[i];

				for(int k = 0; k < layer.neurons.size(); k++) { // loop each neuron

					Neuron& neuron = layer.neurons[k];

					if(j == (layers.size() - 1)) { // calculate correctly the derivative of the cost over the neuron activation value
						neuron.dev_cost_act = 2 * (neuron.activation - Y[i][k]);
					}
					else {
						neuron.dev_cost_act = 0;
						for(int z = 0; z < layers[j + 1].neurons.size(); z++) { 
					// the derivative of the cost based on the activation is calculated by making the sum of the chain rule for each neuron in the layer 
							neuron.dev_cost_act += layers[j + 1].neurons[z].dev_cost_act * layers[j + 1].d_activation(layers[j + 1].neurons[z].linear) * layers[j + 1].neurons[z].weights[k];
						}
					}

					double d_cost_bias = layer.d_activation(neuron.linear)* neuron.dev_cost_act;
					neuron.sum_bias += d_cost_bias;
					for(int n = 0; n < neuron.weights.size(); n++) {
						neuron.sum_weights[n] += previous_layer[n] * d_cost_bias;
					}

				}

			}

		}

		for(int i = 0; i < layers.size(); i++) {
			for(int j = 0; j < layers[i].neurons.size(); j++) {
				layers[i].neurons[j].apply_wb(lr,X.size());
			}
		}
		
	}
};

int main() {
	srand(time(0));
	vector<double> v = {1,0};
	Net rete({2,2,2},{linear,linear},{d_linear,d_linear});

	vector<vector<double>> X = {{1,0},{0,1}},Y = {{0,1},{1,0}};

	while(1) {
		rete.back_prop(X,Y,0.03);
		cout << rete.net_error(X,Y) << endl;
	}


	cout << endl;

	return 0;
}
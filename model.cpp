#include "dataset.cpp"

/*  
    Usage :
        This code implements an LSTM (Long Short-Term Memory) cell for deep learning, which is primarily used for sequential data processing and prediction tasks. 
        The LSTM class defines key components like the hidden units, output units, and various weights and biases. 
        It includes methods for initializing parameters, performing forward and backward propagation, calculating loss and accuracy, and training the model using the Adam optimizer. 
        The model processes input data in batches and updates weights through backpropagation. The predict method allows generating predictions on new data using the trained model.
*/

// Activation Functions
Tensor2f sigmoid(Tensor2f X){
    return X.sigmoid();
}
Tensor2f tanh(Tensor2f X){
    return X.tanh();
}
Tensor2f softmax(Tensor2f X){
    Tensor2f exp_X = X.exp();
    Tensor2f exp_sum_X = exp_X.sum(Eigen :: array<Eigen :: Index , 1>{1}).reshape(Eigen :: array<Eigen :: Index , 2> {exp_X.dimension(0) , 1});
    return exp_X / exp_sum_X.broadcast(Eigen :: array<Eigen :: Index , 2> {1 , X.dimension(1)});
}

class LSTM{
    Tensor2f matmul(Tensor2f a , Tensor2f b){
            return a.contract(b , Eigen :: array < Eigen :: IndexPair<int> , 1> {{Eigen :: IndexPair<int> (1 , 0)}});
    }
    void set_weights(Tensor2f& w , float mean = 0){
        float std = sqrt(2.0 / w.dimension(1));
        w.setRandom();
        w = w * std + mean;
    }
    public:
        int hidden_units;
        int embed_units;
        int output_units;
        int vocab_size;
        unordered_map<string, Tensor2f> V;
        unordered_map<string, Tensor2f> S;
        unordered_map<string, Tensor2f> parameters;
        Tensor2f embedding_weights;
        Tensor2f embedding_bias;

        LSTM(int hidden_units = 256, int output_units = 27 , int vocab_size = 27, int embed_units = 50){
            this->vocab_size = vocab_size;
            this->output_units = output_units;
            this->hidden_units = hidden_units;
            this->embed_units = embed_units;
        }

        void initialize_parameters(){
            Tensor2f WForget(this->hidden_units + this->embed_units , this->hidden_units);
            Tensor2f WInput(this->hidden_units + this->embed_units , this->hidden_units);
            Tensor2f WGate(this->hidden_units + this->embed_units , this->hidden_units);
            Tensor2f WOutput(this->hidden_units + this->embed_units , this->hidden_units);
            Tensor2f WPred(this->hidden_units , this->output_units);
            Tensor2f WEmbeddings(this->vocab_size , this->embed_units);
            
            set_weights(WForget); set_weights(WInput); set_weights(WGate); set_weights(WOutput); set_weights(WPred) ; set_weights(WEmbeddings);

            Tensor2f Bf(1 , this->hidden_units); Bf.setZero();
            Tensor2f Bi(1 , this->hidden_units); Bi.setZero();
            Tensor2f Bg(1 , this->hidden_units); Bg.setZero();
            Tensor2f Bo(1 , this->hidden_units); Bo.setZero();
            Tensor2f Bp(1 , this->output_units); Bp.setZero();
            Tensor2f BEmbeddings(1 , this->embed_units); BEmbeddings.setZero();

            this->parameters["wf"] = WForget;
            this->parameters["wi"] = WInput;
            this->parameters["wg"] = WGate;
            this->parameters["wo"] = WOutput;
            this->parameters["wp"] = WPred;
            this->embedding_weights = WEmbeddings;

            this->parameters["bf"] = Bf;
            this->parameters["bi"] = Bi;
            this->parameters["bg"] = Bg;
            this->parameters["bo"] = Bo;
            this->parameters["bp"] = Bp;
            this->embedding_bias = BEmbeddings;
        }

        pair<unordered_map<string , Tensor2f> , vector<Tensor2f>> one_step_forward_propagation(Tensor2f batch_slice, Tensor2f prev_hidden_state , Tensor2f prev_cell_state){
            auto broadcast = [](const Tensor2f mat , int m) -> Tensor2f{
                return mat.broadcast(Eigen :: array <Eigen :: Index , 2> {m , 1});
            };
            
            int m = batch_slice.dimension(0);
            Tensor2f embedding_slice = matmul(batch_slice , this->embedding_weights) + broadcast(this->embedding_bias , m);

            Tensor2f concat = embedding_slice.concatenate(prev_hidden_state , 1);

            Tensor2f fgo = matmul(concat , this->parameters["wf"]) + broadcast(this->parameters["bf"] , m);
            Tensor2f forget_activation = sigmoid(fgo);

            Tensor2f igo = matmul(concat , this->parameters["wi"]) + broadcast(this->parameters["bi"] , m);
            Tensor2f input_activation = sigmoid(igo);

            Tensor2f ggo = matmul(concat , this->parameters["wg"]) + broadcast(this->parameters["bg"] , m);
            Tensor2f gate_activation = tanh(ggo);

            Tensor2f ogo = matmul(concat , this->parameters["wo"]) + broadcast(this->parameters["bo"] , m);
            Tensor2f output_activation = sigmoid(ogo);

            Tensor2f cell_state = prev_cell_state * forget_activation + (input_activation * gate_activation);

            Tensor2f hidden_state = tanh(cell_state) * output_activation;
            Tensor2f out = matmul(hidden_state , this->parameters["wp"]) + broadcast(this->parameters["bp"] , m);
            Tensor2f prediction = softmax(out);

            vector<Tensor2f> lstm_cell_outputs = {prediction , hidden_state , cell_state};
            unordered_map<string, Tensor2f> cache;
            cache["hidden_state"] = hidden_state;
            cache["cell_state"] = cell_state;
            cache["prev_hidden_state"] = prev_hidden_state;
            cache["prev_cell_state"] = prev_cell_state;
            cache["forget_activation"] = forget_activation;
            cache["input_activation"] = input_activation;
            cache["gate_activation"] = gate_activation;
            cache["output_activation"] = output_activation;
            cache["concat"] = concat;
            cache["embedding_slice"] = embedding_slice;

            return {cache , lstm_cell_outputs};
            
        }

        pair<vector<unordered_map<string , Tensor2f>> , vector<vector<Tensor2f>>> forward_propagation(Tensor3f batch , Tensor2f hidden_state){
            vector<Tensor2f> all_hidden_states;
            vector<Tensor2f> all_cell_states;
            vector<Tensor2f> all_predictions;
            vector<unordered_map<string , Tensor2f>> all_caches;

            Tensor2f cell_state(batch.dimension(0) , this->hidden_units); 
            cell_state.setZero();

            for(int i=0; i < batch.dimension(1)-1; i++){
                // Slicing
                Tensor2f slice = batch.chip(i , 1);
                vector<Tensor2f> lstm_cell_outputs;
                auto output = one_step_forward_propagation(slice , hidden_state , cell_state);
                lstm_cell_outputs = output.second;

                all_caches.push_back(output.first);
                all_predictions.push_back(lstm_cell_outputs[0]);

                hidden_state = lstm_cell_outputs[1];
                cell_state = lstm_cell_outputs[2];
                all_hidden_states.push_back(hidden_state);
                all_cell_states.push_back(cell_state);
            }
            vector<vector<Tensor2f>> forward_output = {all_hidden_states , all_predictions , all_cell_states};
            return {all_caches , forward_output};
        }
        
        pair<float , float> loss_acc_calculation(vector<Tensor2f> outputs , Tensor3f originals){
            // loss = catergorical crossentropy
            int m = originals.dimension(0);
            Tensor2f loss(m , 1);
            loss.setZero();
            Tensor2f accuracy(m , 1);
            accuracy.setZero();

            for(int i=1; i<outputs.size()+1 ; i++){
                Tensor2f original_slice = originals.chip(i , 1);
                Tensor2f log_outputs = outputs[i-1].log();

                loss += (original_slice * log_outputs).sum(Eigen :: array<Eigen :: Index , 1>{1}).reshape(Eigen :: array<Eigen :: Index , 2> {original_slice.dimension(0) , 1});

                Tensor2f temp = (original_slice.argmax(1) == outputs[i-1].argmax(1)).cast<float>().reshape(Eigen::array<Eigen :: Index, 2> {m , 1});
                accuracy += temp;
            }
            float total_loss = 0.0;
            float total_accuracy = 0.0;

            for(int i=0; i<m; i++){
                total_loss += loss(i , 0) / outputs.size();
                total_accuracy += accuracy(i , 0) / outputs.size();
            }
            return {-1 * total_loss / m , total_accuracy / m};
        }

        unordered_map<string , Tensor2f> initialize_gradients(){
            Tensor2f dWp(this->hidden_units , this->output_units); dWp.setZero();
            Tensor2f dWo(this->hidden_units + this->embed_units , this->hidden_units); dWo.setZero();
            Tensor2f dWg(this->hidden_units + this->embed_units , this->hidden_units); dWg.setZero();
            Tensor2f dWi(this->hidden_units + this->embed_units , this->hidden_units); dWi.setZero();
            Tensor2f dWf(this->hidden_units + this->embed_units , this->hidden_units); dWf.setZero();
            Tensor2f dEmbedding(this->vocab_size , this->embed_units); dEmbedding.setZero();
            
            Tensor2f Bf(1 , this->hidden_units); Bf.setZero();
            Tensor2f Bi(1 , this->hidden_units); Bi.setZero();
            Tensor2f Bg(1 , this->hidden_units); Bg.setZero();
            Tensor2f Bo(1 , this->hidden_units); Bo.setZero();
            Tensor2f Bp(1 , this->output_units); Bp.setZero();
            Tensor2f BEmbeddings(1 , this->embed_units); BEmbeddings.setZero();

            unordered_map<string , Tensor2f> gradients;
            gradients["dbf"] = Bf; gradients["dbi"] = Bi; gradients["dbg"] = Bg; gradients["dbo"] = Bo; gradients["dbp"] = Bp; gradients["dbe"] = BEmbeddings;
            gradients["dwe"] = dEmbedding ; gradients["dwp"] = dWp; gradients["dwo"] = dWo; gradients["dwg"] = dWg; gradients["dwi"] = dWi; gradients["dwf"] = dWf;
            return gradients;
        }

        unordered_map<string , Tensor2f> back_propagation(vector<Tensor2f> outputs , Tensor3f originals , vector<unordered_map<string , Tensor2f>> cache){
            int m = originals.dimension(0);
            unordered_map<string , Tensor2f> gradients = initialize_gradients();
            Tensor2f dc_next(m , this->hidden_units); dc_next.setZero();

            for(int i = outputs.size() - 1; i >= 0; i--){
                Tensor2f original_output = originals.chip(i + 1 , 1);
                gradients["dwp"] += matmul(cache[i]["hidden_state"].shuffle(Eigen :: array <Eigen :: Index , 2> {1 , 0}) , original_output - outputs[i]);
                gradients["dbp"] += gradients["dwp"].sum(Eigen :: array <int , 1> {0}).reshape(Eigen :: array <Eigen :: Index , 2> {1 , gradients["dwp"].dimension(1)});
 
                Tensor2f da_next = matmul( original_output - outputs[i] , this->parameters["wp"].shuffle(Eigen :: array <int , 2> {1 , 0}));

                Tensor2f dot = da_next * tanh(cache[i]["cell_state"]) * cache[i]["output_activation"] * (1 - cache[i]["output_activation"]);
                Tensor2f dgt = ((dc_next * cache[i]["input_activation"] + cache[i]["output_activation"] * (1 - tanh(cache[i]["cell_state"]).square())) * cache[i]["input_activation"] * da_next) * (1 - cache[i]["gate_activation"].square());
                Tensor2f dit = ((dc_next * cache[i]["gate_activation"] + cache[i]["output_activation"] * (1 - tanh(cache[i]["cell_state"]).square())) * cache[i]["gate_activation"] * da_next) * cache[i]["input_activation"] *(1 - cache[i]["input_activation"]);
                Tensor2f dft = ((dc_next * cache[i]["prev_cell_state"] + cache[i]["output_activation"] * (1 - tanh(cache[i]["cell_state"]).square())) * cache[i]["prev_cell_state"] * da_next) * cache[i]["forget_activation"] *(1 - cache[i]["forget_activation"]);

                gradients["dwo"] += matmul(cache[i]["concat"].shuffle(Eigen :: array<int , 2> {1 , 0}) , dot);
                gradients["dwg"] += matmul(cache[i]["concat"].shuffle(Eigen :: array<int , 2> {1 , 0}) , dgt);
                gradients["dwi"] += matmul(cache[i]["concat"].shuffle(Eigen :: array<int , 2> {1 , 0}) , dit);
                gradients["dwf"] += matmul(cache[i]["concat"].shuffle(Eigen :: array<int , 2> {1 , 0}) , dft);

                gradients["dbo"] += gradients["dwo"].sum(Eigen :: array <int , 1> {0}).reshape(Eigen :: array <Eigen :: Index , 2> {1 , gradients["dwo"].dimension(1)});
                gradients["dbg"] += gradients["dwg"].sum(Eigen :: array <int , 1> {0}).reshape(Eigen :: array <Eigen :: Index , 2> {1 , gradients["dwg"].dimension(1)});
                gradients["dbi"] += gradients["dwi"].sum(Eigen :: array <int , 1> {0}).reshape(Eigen :: array <Eigen :: Index , 2> {1 , gradients["dwi"].dimension(1)});
                gradients["dbf"] += gradients["dwf"].sum(Eigen :: array <int , 1> {0}).reshape(Eigen :: array <Eigen :: Index , 2> {1 , gradients["dwf"].dimension(1)});

                dc_next = (dc_next * cache[i]["forget_activation"] + cache[i]["output_activation"] * (1 - tanh(cache[i]["cell_state"]).square())) * cache[i]["forget_activation"] * da_next;
                auto slicer = [](Tensor2f& matrix , int embed_units) -> Tensor2f{
                    Eigen :: array <Eigen :: Index , 2> start = {0 , 0};
                    Eigen :: array <Eigen :: Index , 2> end = {embed_units ,matrix.dimension(1)};
                    return matrix.slice(start , end).shuffle(Eigen :: array <int , 2> {1 , 0});
                };
                Tensor2f dembedt = matmul(dft , slicer(this->parameters["wf"] , this->embed_units)) + matmul(dit , slicer(this->parameters["wi"] , this->embed_units)) + matmul(dgt , slicer(this->parameters["wg"] , this->embed_units)) + matmul(dot , slicer(this->parameters["wo"] , this->embed_units));
                Tensor2f input_slice = originals.chip(i , 1);
                gradients["dwe"] += matmul(input_slice.shuffle(Eigen :: array<int , 2> {1 , 0}) , dembedt);
                gradients["dbe"] += gradients["dwe"].sum(Eigen :: array <int , 1> {0}).reshape(Eigen :: array <Eigen :: Index , 2> {1 , gradients["dwe"].dimension(1)});
            }

            return gradients;
        }

        // Adam Optimizer
        void initialize_optimization_parameters(){
            Tensor2f Wp(this->hidden_units , this->output_units); Wp.setZero();
            Tensor2f Wo(this->hidden_units + this->embed_units , this->hidden_units); Wo.setZero();
            Tensor2f Wg(this->hidden_units + this->embed_units , this->hidden_units); Wg.setZero();
            Tensor2f Wi(this->hidden_units + this->embed_units , this->hidden_units); Wi.setZero();
            Tensor2f Wf(this->hidden_units + this->embed_units , this->hidden_units); Wf.setZero();
            Tensor2f We(this->vocab_size , this->embed_units); We.setZero();

            this->V["wp"] = Wp; this->V["wo"] = Wo; this->V["wg"] = Wg; this->V["wi"] = Wi; this->V["wf"] = Wf; this->V["we"] = We;
            this->S["wp"] = Wp.eval(); this->S["wo"] = Wo.eval(); this->S["wg"] = Wg.eval(); this->S["wi"] = Wi.eval(); this->S["wf"] = Wf.eval(); this->S["we"] = We.eval();

            Tensor2f Bf(1 , this->hidden_units); Bf.setZero();
            Tensor2f Bi(1 , this->hidden_units); Bi.setZero();
            Tensor2f Bg(1 , this->hidden_units); Bg.setZero();
            Tensor2f Bo(1 , this->hidden_units); Bo.setZero();
            Tensor2f Bp(1 , this->output_units); Bp.setZero();
            Tensor2f BEmbeddings(1 , this->embed_units); BEmbeddings.setZero();

            this->V["bf"] = Bf; this->V["bi"] = Bi; this->V["bg"] = Bg; this->V["bo"] = Bo; this->V["bp"] = Bp;
            this->V["be"] = BEmbeddings;
            this->S["bf"] = Bf.eval(); this->S["bi"] = Bi.eval(); this->S["bg"] = Bg.eval(); this->S["bo"] = Bo.eval(); this->S["bp"] = Bp.eval();
            this->S["be"] = BEmbeddings.eval();
        }
        
        void optimization(unordered_map<string , Tensor2f> gradients , int epoch, float learning_rate = 0.001, float beta1 = 0.9, float beta2 = 0.999 , float epsilon = 1e-7){
            vector<string> temp = {"wp" , "wo" , "wg" , "wi" , "wf" , "bp" , "bo" , "bg" , "bi" , "bf"};

            for(string wt : temp){
                this->V[wt] = this->V[wt] * beta1 - (1 - beta1) * gradients["d" + wt]; 
                this->S[wt] = this->S[wt] * beta2 + (1 - beta2) * (gradients["d" + wt]).square();

                Tensor2f V_corrected = this->V[wt] / float(1 - pow(beta1 , epoch));
                Tensor2f S_corrected = this->S[wt] / float(1 - pow(beta2 , epoch));

                this->parameters[wt] = this->parameters[wt] - learning_rate * V_corrected / (S_corrected.sqrt() + epsilon);
            }
            // For embeddings weights 
            this->V["we"] = this->V["we"] * beta1 - (1 - beta1) * (gradients["dwe"]);
            this->S["we"] = this->S["we"] * beta2 + (1 - beta2) * (gradients["dwe"]).square();

            Tensor2f V_corrected_weight = this->V["we"] / float(1 - pow(beta1 , epoch));
            Tensor2f S_corrected_weight = this->S["we"] / float(1 - pow(beta2 , epoch));

            this->embedding_weights = this->embedding_weights - learning_rate * V_corrected_weight / (S_corrected_weight.sqrt() + epsilon);
            
            // For embedding bias
            this->V["be"] = this->V["be"] * beta1 - (1 - beta1) * (gradients["dbe"]);
            this->S["be"] = this->S["be"] * beta2 + (1 - beta2) * (gradients["dbe"]).square();

            Tensor2f V_corrected_bias = this->V["be"] / float(1 - pow(beta1 , epoch));
            Tensor2f S_corrected_bias = this->S["be"] / float(1 - pow(beta2 , epoch));

            this->embedding_bias = this->embedding_bias - learning_rate * V_corrected_bias / (S_corrected_bias.sqrt() + epsilon);
        }

        vector<vector<float>> train(Tensor3f dataset , int epochs = 30 , int batch_size = 32){
            int m = dataset.dimension(0);
            int time_steps = dataset.dimension(1);
            initialize_parameters();
            initialize_optimization_parameters();
            float total_loss = 0.0;
            float total_accuracy = 0.0;
            vector<vector<float>> result;

            for(int i=0; i<epochs; i++){
                int no_of_iterations = m / batch_size;
                float loss = 0.0;
                float accuracy = 0.0;
                cout << "Epoch-" << i + 1 << "/" << epochs << " :" << endl;
                for(int j=0; j<no_of_iterations; j++){
                    Eigen :: array<Eigen :: Index , 3> start = {j*batch_size , 0 , 0};
                    Eigen :: array<Eigen :: Index , 3> extent = {min(batch_size , m-j*batch_size), dataset.dimension(1) , dataset.dimension(2)};
                    Tensor3f batch = dataset.slice(start , extent);

                    Tensor2f hidden_state(batch.dimension(0) , this->hidden_units);
                    vector<unordered_map<string , Tensor2f>> cache;
                    vector<Tensor2f> outputs;
                    auto forward_pass_output = forward_propagation(batch , hidden_state);
                    cache = forward_pass_output.first;
                    outputs = forward_pass_output.second[1];

                    auto loss_metric_output = loss_acc_calculation(outputs , batch);
                    loss += loss_metric_output.first;
                    accuracy += loss_metric_output.second;

                    unordered_map<string , Tensor2f> gradients = back_propagation(outputs , batch , cache);
                    optimization(gradients, i + 1);
                    vector<float> metric = {loss / (j+1) , accuracy / (j+1)};
                    result.push_back(metric);
                    cout << "\r[" << j << "/" << no_of_iterations << "] "<< "[" << string((j * 20) / no_of_iterations, '=') << ">" << string(20 - (j * 20) / no_of_iterations, '-') << "]"<< " Loss: " << fixed << setprecision(4) << loss / (j + 1) << " Accuracy: " << fixed << setprecision(4) << accuracy / (j + 1) << flush;
                }
                total_loss += loss / no_of_iterations;
                total_accuracy += accuracy / no_of_iterations;
                cout << endl;
            }
            vector<float> final_metrics = {total_loss / epochs , total_accuracy / epochs};
            result.push_back(final_metrics);
            return result;
        }
        vector<Tensor2f> predict(Tensor3f batch){
            Tensor2f hidden_state(batch.dimension(0) , this->hidden_units); 
            hidden_state.setZero();

            Tensor2f cell_state(batch.dimension(0) , this->hidden_units); 
            cell_state.setZero();

            vector<Tensor2f> result;
            for(int i=0; i < batch.dimension(1)-1; i++){
                Tensor2f slice = batch.chip(i , 1);
                vector<Tensor2f> lstm_cell_outputs;
                auto output = one_step_forward_propagation(slice , hidden_state , cell_state);
                lstm_cell_outputs = output.second;
                hidden_state = lstm_cell_outputs[1];
                cell_state = lstm_cell_outputs[2];

                Tensor2f prediction = lstm_cell_outputs[0];
                result.push_back(prediction);
            }
            return result;
        }
        unordered_map<string , Tensor2f> get_weights(){
            this->parameters["we"] = this->embedding_weights;
            this->parameters["be"] = this->embedding_bias;
            return this->parameters;
        }

        void set_weights(unordered_map<string , Tensor2f> params){
            this->embedding_weights = params["we"];
            this->embedding_bias = params["be"];

            params.erase("we"); params.erase("be");
            this->parameters = params;
        }
};
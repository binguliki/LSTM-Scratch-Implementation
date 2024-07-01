#include <vector>
#include "matplotlibcpp.h"
#include "model.cpp"
using namespace std;
namespace plt = matplotlibcpp;

/*  
    Usage:
        This C++ code snippet demonstrates loading a dataset, training an LSTM model, visualizing training metrics, and making predictions. 
        Key functions include `convert_json_to_weights` for reading model weights from a JSON file, `txt_to_metrics` for reading training metrics from a text file, and `plot_training` for visualizing loss and accuracy with Matplotlib. 
        The `main` function initializes the dataset and model, loads pre-trained weights, visualizes training metrics, and performs predictions on a test dataset. 
        Paths for the dataset, weights, and metrics should be specified by replacing placeholders in the code.
*/
unordered_map<string, Tensor2f> convert_json_to_weights(const string& path) {
    unordered_map<string, Tensor2f> weights;

    ifstream file(path);
    string line;
    string key;
    vector<vector<float>> values;
    auto process_line = [](string line) -> vector<float>{
            if(line.back() == ','){
                line.pop_back();
            }
            string trimmed_line = line.substr(1, line.length() - 1);
            stringstream ss(trimmed_line);
            float value;
            char comma;
            vector<float> values;
            while (ss >> value) {
                values.push_back(value);
                if (!(ss >> comma)){
                    break; 
                }
            }
            return values;
    };

    while(getline(file , line)){
        if(line.length() == 2 || line == "]"){ // Closing bracket condition "]," or "]"
            Tensor2f tensor(values.size(), values[0].size());
            for (int i = 0; i < values.size(); ++i) {
                for (int j = 0; j < values[i].size(); ++j) {
                    tensor(i, j) = values[i][j];
                }
            }
            values.clear();
            weights[key] = tensor;
        }else if(line.length() == 6){ // Key condition
            key = line.substr(1 , 2);
        }else if(line.length() > 6){ // Values condition
            vector<float> row = process_line(line);
            values.push_back(process_line(line));
        }else{ // Json end condition
            continue; 
        }
    }
    return weights;
}
pair<vector<double>, vector<double>> txt_to_metrics(string path) {
    ifstream File(path);
    string line;
    vector<double> loss;
    vector<double> accuracy;

    while (getline(File, line)) {
        stringstream ss(line);
        double loss_value, accuracy_value;
        ss >> loss_value >> accuracy_value;
        loss.push_back(loss_value);
        accuracy.push_back(accuracy_value);
    }

    return {loss, accuracy};
}
void plot_training(vector<double> loss , vector<double> accuracy){
    plt::figure();
    plt::plot(loss, {{"label" , "Loss"}});
    plt::plot(accuracy , {{"label" , "Accuracy"}});
    plt::ylim(0 , 2);
    plt::legend();
    plt::title("Performance");
    plt::xlabel("Iterations");
    plt::save("performance.png");
    plt::show();
}

int main() {
    string dataset_path = ""; // Provide the dataset path
    Dataset* processor = new Dataset(dataset_path);
    Tensor3f dataset = processor->preprocess();
    LSTM model = LSTM();
    string weights_path = ""; // Provide path to weights.json file

    unordered_map<string , Tensor2f> weights = convert_json_to_weights(weights_path);
    model.set_weights(weights);

    string metrics_path = ""; // Provide path to metrics.txt file
    auto training_metrics = txt_to_metrics(metrics_path);
    plot_training(training_metrics.first , training_metrics.second);

    // Testing over 2000 instances
    Eigen :: array<Eigen :: Index , 3> start = {12001 , 0 , 0};
    Eigen :: array<Eigen :: Index , 3> extent = {1999 , dataset.dimension(1) , dataset.dimension(2)};
    Tensor3f batch = dataset.slice(start , extent);

    cout << "Prediction... "<< endl;
    vector<Tensor2f> temp = model.predict(batch); // Each row is a timestep
    Tensor3f prediction(temp[0].dimension(0), temp.size(), temp[0].dimension(1)); prediction.setZero(); 
    for(int i = 0; i < temp.size(); i++){
        prediction.chip(i, 1) = temp[i];
    }
    
    vector<string> originals = processor->convert_data_to_string(batch);
    vector<string> names = processor->convert_data_to_string(prediction);
    
    auto metrics = model.loss_acc_calculation(temp , batch);
    cout << "Test Accuracy : " << metrics.second << endl;

    delete processor;

}
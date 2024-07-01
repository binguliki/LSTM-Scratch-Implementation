#include "model.cpp"
using namespace std;

/*  
    Usage:
        The given C++ code snippet demonstrates the process of loading a dataset, preprocessing the data, training an LSTM model, and subsequently saving the metrics and weights to files. 
        Key functions include convert_weights_to_json, which converts the model weights into JSON format and saves them to a specified file, and convert_metrics_to_txt, which saves the training metrics to a text file. 
        The main function initializes the dataset and model, performs training, and invokes these conversion functions to store the results. 
        Paths for saving the weights and metrics should be specified by replacing the placeholders within the code. 
*/
void convert_weights_to_json(unordered_map<string , Tensor2f> weights){
    string path = ""; // Replace the path with directory where you want to store weights
    ofstream file(path);
    file << "{\n";
    bool first = true;
    for (const auto& pair : weights) {
        if (!first) {
            file << ",\n";
        }
        file << '"' << pair.first << '"' << ":[\n";
        for (int i = 0; i < pair.second.dimension(0); ++i) {
            if (i > 0) {
                file << ",\n";
            }
            file << "[";
            for (int j = 0; j < pair.second.dimension(1); ++j) {
                if (j > 0) {
                    file << ",";
                }
                file << pair.second(i, j);
            }
            file << "]";
        }
        file << "\n]";
        first = false;
    }
    file << "\n}\n";
    file.close();
}
void convert_metrics_to_txt(vector<vector<float>> metrics){
    int length = metrics.size();
    string path = ""; // Replace the path with directory where you want to store metrics
    ofstream file(path);
    for(int i=0 ; i<length; i++){
        file << metrics[i][0] << " " << metrics[i][1] << "\n";
    }
    file.close();
}
int main(){
    string path = "./data/names.txt";
    Dataset* processor = new Dataset(path);
    Tensor3f train_data = processor->preprocess();
    
    // Training over first 2000 instances
    Eigen :: array<Eigen :: Index , 3> start = {0 , 0 , 0};
    Eigen :: array<Eigen :: Index , 3> extent = {12000 , train_data.dimension(1) , train_data.dimension(2)};
    train_data = train_data.slice(start , extent);

    cout << "Data dimensions : " << train_data.dimensions() << endl;
    cout << string(20 , '-') << endl;
    LSTM model = LSTM(); 
    cout << "Training ..." << endl;

    vector<vector<float>> metrics = model.train(train_data , 10 , 128);
    cout << "Final Loss : " << metrics.back()[0] << endl;
    cout << "Final Accuracy : " << metrics.back()[1] << endl;

    metrics.pop_back(); // Removing Final metrics
    convert_metrics_to_txt(metrics);
    convert_weights_to_json(model.get_weights());
    return 0;
}
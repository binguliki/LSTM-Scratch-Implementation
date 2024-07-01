#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <map>
#include <algorithm>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
using namespace std;

/*  Usage :
        The provided `Dataset` class handles text data for a character-level generation model. 
        It reads data from a file, converts characters to indices using a one-hot encoding scheme, and preprocesses the data into a 3D tensor. 
        This tensor represents each word as a sequence of one-hot encoded vectors. 
        The class also includes a method to convert the tensor back to strings by decoding the one-hot vectors. 
        It supports ASCII characters and a special character for padding. 
        The class ensures all words are transformed to lowercase and padded to a maximum length, facilitating consistent input for training the generation model.
*/
typedef Eigen :: Tensor<float , 2> Tensor2f;
typedef Eigen :: Tensor<float , 3> Tensor3f;
typedef Eigen :: Tensor<float , 1> Tensor1f;

class Dataset{
    public:
        string path;
        vector<string> data;
        int max_length = 27;
        unordered_map <string , int> char_to_idx;
        unordered_map <int , string> idx_to_char;
        Tensor3f processed_data;

        Dataset(string path){
            if(filesystem :: exists(path)){
                this->path = path;
                this->set_data();
            }else{
                cout << "File Not Found !!" << endl;
            }
        }
        void set_data(){
            string line;
            ifstream File(this->path);
            while (getline(File, line)){
                this->data.push_back(line);
            }
        }
        Tensor3f preprocess(){
            if(this->processed_data.size()){
                return this->processed_data;
            }
            for(int i=0; i<26; i++){
                char_to_idx[string(1 , static_cast<char>(97 + i))] = i;
                idx_to_char[i] = static_cast<char>(97 + i);
            }
            char_to_idx["."] = 26;
            idx_to_char[26] = ".";
            
            for(int i=0; i<this->get_length(); i++){
                transform(this->data[i].begin() , this->data[i].end() , this->data[i].begin() , [](char c){return tolower(c);});
                this->data[i] += string(this->max_length - this->data[i].length() , '.');
            }
            Tensor3f matrix(this->get_length() , this->max_length , this->max_length);
            matrix.setZero();

            int idx = 0;
            for(string word : this->data){
                for(int i=0; i<word.length(); i++){
                    matrix(idx , i , char_to_idx[string(1 , word[i])]) = 1.0; 
                }
                idx += 1;
            }

            this->processed_data = matrix;
            return this->processed_data;
        }
        vector<string> convert_data_to_string(Tensor3f matrix){
            vector<string> result;
            for(int i=0; i<matrix.dimension(0); i++){
                string temp = "";
                for(int j=0; j<matrix.dimension(1); j++){
                    Tensor1f word = matrix.chip(i , 0).chip(j , 0);
                    Eigen :: Index max_index = 0;
                    float max_value = word(0);
                    for(int i=1; i<word.dimension(0); i++){
                        if(word(i) > max_value){
                            max_index = i; 
                            max_value = word(i);
                        }
                    }
                    temp += this->idx_to_char[static_cast<int> (max_index)];
                    if(temp.back() == '.'){
                        break;
                    }
                }
                result.push_back(temp);
            }
            return result;
        }
        int get_length(){
            return this->data.size();
        }
};


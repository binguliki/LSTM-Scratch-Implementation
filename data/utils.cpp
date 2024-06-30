#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>

/* Usage:
    This C++ program extracts a specified number of names from a CSV file and writes them to a text file. 
    It allows the user to input how many names to extract, reads these names from the CSV, and saves them into a separate text file for further use.
*/
using namespace std;
int main(){
    string path = "./data/NationalNames.csv";
    ifstream InFile(path);

    int NUM_OF_NAMES;
    cout << "Enter the number of names that has to be included : ";
    cin >> NUM_OF_NAMES;

    string line = "";
    vector<string> names;
    for(int i=0 ; i<=NUM_OF_NAMES; i++){
        getline(InFile , line);
        string name;
        istringstream LINE(line);
        for(int j=0; j<=1; j++){
            getline(LINE , name , ',');
            if(j == 1){names.push_back(name);}
        }
    }  
    InFile.close();

    cout << "Writing ..." << endl;
    string destination_path = "./data/names.txt";
    ofstream OutFile(destination_path);
    for(int i=1; i<=NUM_OF_NAMES; i++){
        OutFile << names[i] + "\n";
    }
    OutFile.close();
    cout << "--> Dumped in " << destination_path << endl;
}
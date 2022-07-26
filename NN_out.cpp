#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h> 
#include <chrono>


using namespace std;

// activation function
double activation(double x)
{
    return 1.0/(1.0 + exp(-x));
}

// derivative of activation function
double d_activation(double x){
    return (1.0 - activation(x))*activation(x);
 }  
 

// 
void Randomize(double* a, int n, double min, double max){
	//cout<<"RAND_MAX="<<RAND_MAX<<endl;
    srand (time(NULL));
    for ( int i = 0 ; i < n ; i++){
        double f = (double)rand() / RAND_MAX;
        a[i]= min + f * (max - min);
    }
}

void PrintVector(double* a, int n){
    for ( int i = 0 ; i < n ; i++){
        cout<<a[i]<<" ";
    }
    cout<<endl;
}


class NN{
//	friend class MatrixD;
//	friend class VectorD;
	
	private:
	  int nInputs; // number of inputs
	  int nOutputs; //number of outputs
	  int nHiddenNeurons; // number of neurons in hidden layer
	 
	  // search parameters
	  double dw;   // step for gradient estimation
	  double learningRate; // hm, learning  rate
	  int nStepsMax;
	  
	  // whole training set 
	  double* inputsTraining;   // inputs - column, pattern - row 
	  double* outputsTraining;  // outputs - column, pattern - row
	  int nTrainingEntries;  // number of rows in training set
	  
	  // current NN situation
	  double* currentInputs;  // row picked from inputsTraining
	  double* currentAnswers; // ditto for training answers 
	  double* currentOutputs; // guess what? 
	  double* currentError;   // current differnce between answers and output
	  //output error for current dataset row
	  double sumOfOutputErrors;
	  // sum of errors for all dataset entries
	  double totalDatasetError;
	  
	  
	  // input to hidden layer
	  double* weightsHidden;  // hidden layer weights matrix Wh
	  double* biasHidden;     // hidden layer bias vector bh
	  double* d_weightsHidden;  // hidden layer weights matrix Wh derivative
	  double* d_biasHidden;     // hidden layer bias vector bh derivative
	  double* deltaHidden;  // for backpropagation
	  
	  // state of the hidden layer
	  double* netHidden;     // yh = Wh*z + bh - renaming 
	  double* outHidden;   //  zh = activ(Wh*x + bh) - layer output
	  
      // hidden layer to output layer
      double* weightsOutput;  // Wh
      double* biasOutput;     // bh
      double* d_weightsOutput;  // dWh/dw - gradients
      double* d_biasOutput;     // dbh / dw - gradients
      double* deltaOutput;     // dbh / dw - gradients
 
	  // state of the output layer
      double* nnSumOutput;  
      double* netOutput;  // yo = Wo*zh + bo
      double* nnBiaOutput;
	  double* nnOutOutput; // zo = activ(yo)
	  
	public:
	  NN(){ }; // constructor
	  int LoadTrainingSet(string file,int nInp,int nHid , int nOut);
	  void DisplayDigit(int nImage); // ASCII art display
	  int InitNet(double min, double max);
	  void GetTrainingEntry(int iTrainRow);
	  void ForwardProp();
	  double GetTotalError(){ return sumOfOutputErrors;};
	  void PrintErr(){ cout<<" Error: "<<sumOfOutputErrors<<endl;};
	  void DirectGradientEstimation();
	  void BackProp();
	  void StepByGradient();
	  double GetOutputError(){ return sumOfOutputErrors;};
	  double TotalDatasetError(); // sum of errors for all rows of train data
	  void Train1();
	  void PrintOutputs();
      void DisplayResults();	
};

// loads inputsTraining and outputsTraining from "file"
int NN::LoadTrainingSet(string file,int nInp,int nHid , int nOut){
	std::ifstream data(file);
    std::string line;
    nTrainingEntries = 0;
    nInputs = nInp;
    nOutputs = nOut;
    nHiddenNeurons  = nHid;
    // count number of lines in input file
    while(std::getline(data,line)) { nTrainingEntries++; }
    cout<<" Thera are "<<nTrainingEntries<<" entries in dataset"<<endl;
    // reserve the memory
    inputsTraining = new double[nTrainingEntries*nInputs];
    outputsTraining = new double[nTrainingEntries*nOutputs];
    cout<<" Memory reserved..."<<endl;
    // rewind the file
    data.clear();
    data.seekg(0);
    // read training data file
    for(int iim = 0; iim<nTrainingEntries; iim++) {
		std::getline(data,line);
    	//cout<<" iim= "<<iim<<" Input: "<<line<<endl;
        std::stringstream lineStream(line);
        std::string cell;
        int count = 0;
        // break input string into inputs and answers
        while(std::getline(lineStream,cell,' ')) {
            //cout<<"count="<<count<<"cell="<<cell<<" "<<endl;
            if (count<nInputs) { 
				inputsTraining[iim*nInputs+count] = atof(cell.c_str()) ;
				//cout<<" count="<<count<<" Inp[][]="<<inputsTraining.GetElement(iim,count)<<endl;
			} else {
				outputsTraining[iim*nOutputs+count-nInputs] = atof(cell.c_str());
				//cout<<" count-nInputs="<<count-nInputs<<" Out[][]="<<outputsTraining.GetElement(iim,count-nInputs)<<endl;
			}	
          count++;
        } // while
        //cout<<" Input string "<<iim<<" parsed"<<endl;
        //char stop;
        //cin>>stop; 
    } // for
    cout<<" Training set loaded. Inputs:"<<endl;
    //char stop;
    //cin>>stop; 
    //inputsTraining.PrintMatrix();
    data.close();
	return 0;
}



// reserves the memory and puts
//random values (range min-max) into weights  and biases
int NN::InitNet(double min, double max){
	
	cout<<" InitNet: nInputs="<<nInputs<<" nHiddenNeurons=";
	cout<<nHiddenNeurons<<" nOutputs="<<nOutputs<<endl; 
	// reserve the memory for weights and biases
	// hidden layer
	weightsHidden = new double[nHiddenNeurons*nInputs];
	biasHidden = new double[nHiddenNeurons];
	d_weightsHidden = new double[nHiddenNeurons*nInputs];
	d_biasHidden = new double[nHiddenNeurons];
	deltaHidden = new double[nHiddenNeurons];
	// output layer
	weightsOutput = new double[nHiddenNeurons*nOutputs];
	biasOutput = new double[nOutputs];
	d_weightsOutput = new double[nHiddenNeurons*nOutputs];
	d_biasOutput = new double[nOutputs];
	deltaOutput = new double[nOutputs];
	
	// current input and output vector, answers and error	
	currentInputs = new double[nInputs];
	currentOutputs = new double[nOutputs];
	currentAnswers = new double[nOutputs];
	currentError =  new double[nOutputs];
	
	// reserve memory for current net levels
	netHidden = new double[nHiddenNeurons];
	outHidden = new double[nHiddenNeurons];
	netOutput = new double[nOutputs];
	
	// make weights and biases random
	Randomize(weightsHidden,nHiddenNeurons*nInputs,min,max);
	Randomize(biasHidden,nHiddenNeurons,min,max);
	Randomize(weightsOutput,nHiddenNeurons*nOutputs,min,max);
	Randomize(biasOutput,nOutputs,min,max);
   
  	return 0;
}

// loads row of dataset into the net for estimation
void NN::GetTrainingEntry(int iTrainRow){
	for ( int i = 0 ; i<nInputs;i++)
	   currentInputs[i] = inputsTraining[iTrainRow*nInputs+i];
	for (int i = 0 ; i < nOutputs;i++)   
	  currentAnswers[i]= outputsTraining[iTrainRow*nOutputs+i];
	
}

// display digit on the screen as ASCII
void NN::DisplayDigit(int iImage){
  int scan = 0;
  for (int i = 0 ; i < 8; i++){
    for ( int j = 0 ; j < 8;j++){
       if (inputsTraining[iImage*nInputs + scan] > 0.0){
          cout<<"0";
       } else {
         cout<<"-";
       }
       scan++;
    }
    cout<<endl;
  }
}



// direct calculation of forward propagation
void NN::ForwardProp(){
cout<<" ForwardProp() started"<<endl;	
	//  inputs ->  hidden layer
	// for each neuron in hidden layer
	for ( int hid = 0 ;hid < nHiddenNeurons ; hid++){
		// combine inputs and add bias
        netHidden[hid] = biasHidden[hid]; 
        for (int inp = 0 ; inp < nInputs ; inp++){
			cout<<" hid="<<hid<<" inp="<<inp<<" ind="<<hid*nInputs + inp<<endl;			
			netHidden[hid] = netHidden[hid] + currentInputs[inp]* weightsHidden[hid*nInputs + inp];
	    }
	    outHidden[hid] = activation(netHidden[hid]);
	}	
//int st;
//cin>>st;	
	sumOfOutputErrors = 0.0;
	// for each neuron in output layer
	for ( int out = 0 ; out < nOutputs ; out++){
		// combine hidden and add bias 
		netOutput[out] = biasOutput[out];
		for (int hid = 0 ; hid < nHiddenNeurons ; hid++){
			netOutput[out] = netOutput[out] + outHidden[hid]* weightsOutput[out*nHiddenNeurons+hid];
		}
		currentOutputs[out] = activation(netOutput[out]);
		currentError[out] = currentOutputs[out] - currentAnswers[out];
		sumOfOutputErrors = sumOfOutputErrors + currentError[out]*currentError[out]; 
		//sumOfOuput errors discovered will be implimented later 
	}
}
void NN::StepByGradient(){
	//gradientStep(), called each time to 
	for(int i = 0; i < nHiddenNeurons; i++){
		biasHidden[i] -= learningRate * d_biasHidden[i];//setting the biases correctly
	}
	for(int i = 0; i < nHiddenNeurons*nInputs; i++){
		weightsHidden[i] -= learningRate * d_weightsHidden[i];//setting everything completely
	}
	for(int i = 0; i < nHiddenNeurons*nOutputs; i++){
		weightsOutput[i] -= learningRate * d_weightsOutput[i];
	}
	for(int i = 0; i < nOutputs; i++){
		biasOutput[i] -= learningRate * d_biasOutput[i];
	}
}
// calculate gradient by direct estimation
void NN::DirectGradientEstimation(){
	//calculateGradient() from ex2 
	//with 4 loops for vectors weights, bias, hidden biases and hidden weights
	double e0; 
	double e1;
	double d = 0.1; 
	ForwardProp();
	e0 = sumOfOutputErrors;
	//all biases weights
	for(int i = 0; i < nHiddenNeurons; i++){
		biasHidden[i] += d;//bias hidden doesn't chance just needs it to calculate forwardProp 
		ForwardProp(); 	
		e1 = sumOfOutputErrors; 
		d_biasHidden[i] = (e1-e0)/d; //d_is just array of direvatives
		biasHidden[i] -= d;
	}
	//all hidden weight
	for(int i = 0; i < nHiddenNeurons*nInputs; i++){
		weightsHidden[i] += d; 
		ForwardProp();	
		e1 = sumOfOutputErrors; 
		d_weightsHidden[i] = (e1-e0)/d;
		weightsHidden[i] -= d;
	}
	//all output wieghts
	for(int i = 0; i < nHiddenNeurons*nOutputs; i++){
		weightsOutput[i] += d; 
		ForwardProp();	
		e1 = sumOfOutputErrors; 
		d_weightsOutput[i] = (e1-e0)/d;
		weightsOutput[i] -= d;
	}
	//all output biases
	for(int i = 0; i < nOutputs; i++){
		biasOutput[i] += d; 
		ForwardProp();	
		e1 = sumOfOutputErrors; 
		d_biasOutput[i] = (e1-e0)/d;
		biasOutput[i] -= d;
	}
	
}


// calculate gradients by back-propagation
void NN::BackProp(){

}

// change weights and biases in direction oppposite to gradient,
// scaled by learning rate (which should be negative)


// calculates error for all entries in the dataset
// for current values of weights and biases
double NN::TotalDatasetError(){ // sum of errors for all rows of train data
//cout<<" There are "<<nTrainingEntries<<" rows in the dataset"<<endl;
	totalDatasetError = 0.0;
	for ( int entry = 0 ; entry < nTrainingEntries; entry++){
//cout<<"entry = "<<entry<<endl;		
		GetTrainingEntry(entry);
	    ForwardProp();
	    totalDatasetError = totalDatasetError + GetOutputError();
	}
//cout<<" totalDatasetError/nEntries="<<totalDatasetError/nTrainingEntries<<endl;
	return totalDatasetError; //nTrainingEntries;
}



void NN::Train1(){
	// set net search parameters
	dw = 0.001;  // step to estimate gradient
	learningRate = 0.05;
    //DisplayDigit(iImage);
    InitNet(-0.1,0.1);
    int iImage = 0;
    srand (time(NULL));  // seed random number generator
    int searchStep = 0;
    
    while (( searchStep < 5000) && (TotalDatasetError() > 10.0) ){ 
  	  // pick random entry from training dataset
      iImage = nTrainingEntries*(double)rand() / RAND_MAX;
	  // copy inputs and outputs from training matrix into neural netg
  	  GetTrainingEntry(iImage);
      ForwardProp();
      DirectGradientEstimation();
      BackProp();
      StepByGradient();
      cout<<"step: "<<searchStep;//<<" image: "<<iImage<<" Error for current row:"<<GetOutputError();
      cout<<" Total dataset error: "<< TotalDatasetError()<<endl;
      searchStep++;
    }
     
}

void NN::PrintOutputs(){
	cout<<" Net outputs: ";
	for (int out = 0 ; out < nOutputs ; out++){
		cout<<currentOutputs[out]<<"  ";
	}
	cout<<endl;
}


void NN::DisplayResults(){
	int iImage = -1;
	cout<<" There are "<< nTrainingEntries<<" entries "<<endl;
	cout<<" Enter number of the entry to display"<<endl;
	cin>>iImage;
	while (iImage < nTrainingEntries) {
	  // copy inputs and outputs from big matrix
	  GetTrainingEntry(iImage);
	  ForwardProp();
      DisplayDigit(iImage);
      PrintVector(currentOutputs, nOutputs);
      cin>>iImage;
   }
    
}


int main(){
	NN neuralNet;
	neuralNet.LoadTrainingSet("train.txt",64,128,8);
	neuralNet.DisplayDigit(10);
	
	int imageIndex = 0 ;
	neuralNet.InitNet(-0.1,0.1);
	neuralNet.ForwardProp();
	//auto t7 = std::chrono::system_clock::now();
    //  neuralNet.DirectGradientEstimation(); 
	//auto t8 = chrono::system_clock::now();
	//cout<< " Timing Gradient:"<< chrono::duration_cast<std::chrono::milliseconds>(t8 - t7).count() << " ms\n";
	
	neuralNet.Train1();
	
	while ( imageIndex < 750){
	  cout<<" Enter index of the imag to display"; cin>>imageIndex;
	  if (imageIndex<759){
		   neuralNet.GetTrainingEntry(imageIndex);
	       neuralNet.ForwardProp();
	       //need to use direct graident estimation 
	       //then step by gradient
    	   neuralNet.DisplayDigit(imageIndex);
	       neuralNet.PrintOutputs();
      }
    }
     
}

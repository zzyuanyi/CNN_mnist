#define _CRT_SECURE_NO_WARNINGS
using namespace std;
#include <stdio.h>
#include <vector>
#include<iostream>
#include<ctime>
#include<math.h>
#include<stdlib.h>
#include <cassert>

typedef struct Matrix
{
	int _col_num;
	int _row_num;
	double** ptr;
	Matrix()
	{
		_row_num = 0;
		_col_num = 0;
		ptr = nullptr;
	}
	Matrix(const Matrix& other)
	{
		_row_num = other._row_num;
		_col_num = other._col_num;
		ptr = new double* [_row_num];
		for (int i = 0; i < _row_num; ++i)
		{
			ptr[i] = new double[_col_num];
			for (int j = 0; j < _col_num; ++j)
			{
				ptr[i][j] = other.ptr[i][j];
			}
		}
	}
	Matrix(int x, int y)
	{
		_row_num = x;
		_col_num = y;
		ptr = new double* [x];
		for (int i = 0; i < x; i++)
		{
			ptr[i] = new double[y];
		}
		for (int i = 0; i < x; i++)
		{
			for (int j = 0; j < y; j++)
			{
				ptr[i][j] = 0;
			}
		}
	}
	static Matrix zero(int x, int y)
	{
		Mat zeroMat(x,y);
		for (int i = 0; i < x; i++)
		{
			for (int j = 0; j < y; j++)
			{
				zeroMat.ptr[i][j] = 0;
			}
		}
		return zeroMat;
	}
	Matrix& operator=(const Matrix& other)
	{
		if (this == &other)
		{
			return *this;
		}
		
		for (int i = 0; i < _row_num; ++i)
		{
			delete[] ptr[i];
		}
		delete[] ptr;
		
		_row_num = other._row_num;
		_col_num = other._col_num;
		ptr = new double* [_row_num];
		for (int i = 0; i < _row_num; ++i)
		{
			ptr[i] = new double[_col_num];
			for (int j = 0; j < _col_num; ++j)
			{
				ptr[i][j] = other.ptr[i][j];
			}
		}
		
		return *this;
	}
	
	Matrix clone() const
	{
		return Matrix(*this);
	}
	~Matrix() {
		for (int i = 0; i < _row_num; ++i) {
			delete[] ptr[i];
		}
		delete[] ptr;
	}
}Mat;////////////本代码的核心，该模型绝大多数操作均基于矩阵的运算进行。
Mat Addition(Mat& m1, Mat& m2)
{
	//cout << m1._row_num << " " << m1._col_num << " " << m2._row_num << " " << m2._col_num;
	assert(m1._row_num == m2._row_num && m1._col_num == m2._col_num);
	int row = m1._row_num;
	int col = m1._col_num;
	Mat temp(row, col);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			temp.ptr[i][j] = m1.ptr[i][j] + m2.ptr[i][j];
		}
	}
	return temp;
}///////加法，Mat的一个方法用来计算两个矩阵相同位置加和的结果
Mat Multiple(Mat& input,double x)
{
	int row = input._row_num;
	int col = input._col_num;
	Mat temp(row, col);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			temp.ptr[i][j] = input.ptr[i][j] * x;
		}
	}
	return temp;
}///////////倍乘，将矩阵中的每一个数值乘以传入的数值x
double Sum(Mat& input)
{
	int row = input._row_num;
	int col = input._col_num;
	double temp = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			temp += input.ptr[i][j];
		}
	}
	return temp;
}//////矩阵的一个方法用来求对应矩阵中所有的元素加和的结果
typedef struct Convolutional_layer
{
	int _inwide;
	int _inhigh;
	int _cov_size;
	int _inChannels;
	int _outChannels;
	vector<vector<Mat>> _cov_weight;
	Mat _b;
	bool isFullConnect;
	vector<Mat>_in_ac;
	vector<Mat>_out_ac;
	vector<Mat>_grad;
	
}Covlayer;/////暂且不知道为什么会溢出呢？//////////////卷积层，内包含卷积矩阵以及Relu函数
typedef struct Pooling_layer
{
	int _inwide;
	int _inhigh;
	int _pool_size;
	int _inChannels;
	int _outChannels;
	int _pool_type;
	Mat _b;
	vector<Mat>_out;
	vector<Mat>_grad;
	vector<Mat>max_position;
}Poollayer;////////////////////////池化层，可选池化的方式，具有最大池化以及平均池化两种选择可以操作 。
typedef struct nn_layer
{
	int _in_num;
	int _out_num;
	Mat _weight;
	Mat _b;
	Mat _in_ac;
	Mat _out_ac;
	Mat _grad;
	bool isFullConnect;
}Outlayer;//////////////////////////////全连接层，用于将数值均一化之后进行输出
typedef struct cnn_network
{
	int _layer_num;
	Covlayer C1;
	Poollayer P2;
	Covlayer C3;
	Poollayer P4;
	Outlayer O5;
	Mat e;
	Mat L;
}CNN;//////////////本代码中的模型
typedef struct train_opts
{
	int _numepochs; 
	float _alpha; 
}CNNOpts;///////////////////////训练参数设置


Covlayer initCovLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels)
{
	Covlayer covL;
	
	covL._inhigh = inputHeight;
	covL._inwide = inputWidth;
	covL._cov_size = mapSize;
	
	covL._inChannels = inChannels;
	covL._outChannels = outChannels;
	
	covL.isFullConnect = true;   
	
	
	srand((unsigned)time(NULL));   //设置随机数种子
	for (int i = 0; i < inChannels; i++)   
	{
		vector<Mat> tmp;
		for (int j = 0; j < outChannels; j++)   
		{
			Mat tmpmat(mapSize, mapSize);  
			for (int r = 0; r < mapSize; r++)  
			{
				for (int c = 0; c < mapSize; c++) 
				{
					
					float randnum = (((float)rand() / (float)RAND_MAX) - 0.5) * 2;    //生成-1~1的随机数
					tmpmat.ptr[r][c] = randnum;
				}
			}
			tmp.push_back(tmpmat.clone());
		}
		covL._cov_weight.push_back(tmp);
	}
	
	covL._b= Mat::zero(1, outChannels);  
	
	int outW = inputWidth - mapSize + 1;   
	int outH = inputHeight - mapSize + 1;  
	
	Mat tmpmat2 = Mat::zero(outH, outW);
	for (int i = 0; i < outChannels; i++)
	{
		covL._grad.push_back(tmpmat2.clone());  
		covL._in_ac.push_back(tmpmat2.clone());  
		covL._out_ac.push_back(tmpmat2.clone());  
	}
	
	return covL;   
}////////////////////////////////////////////对于卷积层的初始化，在其中对于卷积层中数值进行随机初始化

Poollayer initPoolLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType)
{
	Poollayer poolL;
	
	poolL._inhigh = inputHeight;   
	poolL._inwide = inputWidth;     
	poolL._pool_size = mapSize;          
	poolL._inChannels = inChannels;     
	poolL._outChannels = outChannels;  
	poolL._pool_type = poolType;         
	
	poolL._b = Mat::zero(1, outChannels);    
	
	int outW = inputWidth / mapSize;   
	int outH = inputHeight / mapSize;
	
	Mat tmpmat = Mat::zero(outH, outW);
	Mat tmpmat1 = Mat::zero(outH, outW);
	for (int i = 0; i < outChannels; i++)
	{
		poolL._grad.push_back(tmpmat.clone()); 
		poolL._out.push_back(tmpmat.clone());  
		poolL.max_position.push_back(tmpmat1.clone());  
	}
	
	return poolL;
}/////////////////////////////////////对于池化层的初始化，在这里默认采取的池化方式为最大池化
Outlayer initOutLayer(int inputNum, int outputNum)
{
	Outlayer outL;
	
	outL._in_num = inputNum;
	outL._out_num = outputNum;
	outL.isFullConnect = true;
	
	outL._b = Mat::zero(1, outputNum);    
	outL._grad = Mat::zero(1, outputNum);
	outL._in_ac = Mat::zero(1, outputNum);
	outL._out_ac = Mat::zero(1, outputNum);
	
	
	outL._weight  = Mat::zero(outputNum, inputNum);  
	srand((unsigned)time(NULL));
	for (int i = 0; i < outputNum; i++)
	{
		for (int j = 0; j < inputNum; j++)
		{
			
			double randnum = (((double)rand() / (double)RAND_MAX) - 0.5) * 2; 
			outL._weight.ptr[i][j] = randnum;
		}
	}
	
	return outL;
}/////////////////////////////////////////对于全连接层的初始化，将其中的权重进行随机初始化

void cnnsetup(CNN& cnn, int inputSize_r, int inputSize_c, int outputSize)   
{
	cnn._layer_num = 5;
	
	//C1层
	int mapSize = 5;
	int inSize_c = inputSize_c;   //28
	int inSize_r = inputSize_r;   //28
	int C1_outChannels = 6;
	cnn.C1 = initCovLayer(inSize_c, inSize_r, mapSize, 1, C1_outChannels);   //卷积层1
	
	//P2层
	inSize_c = inSize_c - cnn.C1._cov_size + 1;  //24
	inSize_r = inSize_r - cnn.C1._cov_size + 1;  //24
	mapSize = 2;
	cnn.P2 = initPoolLayer(inSize_c, inSize_r, mapSize, cnn.C1._outChannels, cnn.C1._outChannels, 1);   //池化层
	
	//C3层
	inSize_c = inSize_c / cnn.P2._pool_size;   //12
	inSize_r = inSize_r / cnn.P2._pool_size;   //12
	mapSize = 5;
	int C3_outChannes = 12;
	cnn.C3 = initCovLayer(inSize_c, inSize_r, mapSize, cnn.P2._outChannels, C3_outChannes);   //卷积层
	
	//P4层
	inSize_c = inSize_c - cnn.C3._cov_size + 1;   //8
	inSize_r = inSize_r - cnn.C3._cov_size + 1;   //8
	mapSize = 2;
	cnn.P4 = initPoolLayer(inSize_c, inSize_r, mapSize, cnn.C3._outChannels, cnn.C3._outChannels, 1);    //池化层
	
	//O5层
	inSize_c = inSize_c / cnn.P4._pool_size;   //4
	inSize_r = inSize_r / cnn.P4._pool_size;   //4
	cnn.O5 = initOutLayer(inSize_c * inSize_r * cnn.P4._outChannels, outputSize);    //输出层
	
	cnn.e = Mat::zero(1, cnn.O5._out_num);   //输出层的输出值与标签值之差
}/////////////////////////////////对于网络的初始化函数，调用网络中各层的初始化函数。

Mat Convolution(Mat input, Mat cov_kernel)
{
	int cov_row = cov_kernel._row_num;
	int cov_col = cov_kernel._col_num;
	int in_row = input._row_num;
	int in_col = input._col_num;
	int out_row = in_row - cov_row + 1;
	int out_col = in_col - cov_col + 1;
	Mat output = Mat::zero(out_row, out_col);
	double sum;	
	for (int i = 0; i < out_row; i++)
	{
		for (int j = 0; j < out_col; j++)
		{
			sum = 0;
			for (int q = 0; q < cov_row; q++)
			{
				for (int k = 0; k < cov_col; k++)
				{
					sum += cov_kernel.ptr[q][k] * input.ptr[i + q][j + k];
				}
			}
			output.ptr[i][j] = sum;
			//cout << sum << " ";
		}
	}
	//for (int e = 0; e < out_row; e++)
	//{
	//	for (int t = 0; t < out_col; t++)
	//	{
	//		cout << output.ptr[e][t] << " ";
	//	}
	//	cout << "\n";
	//}
	return output;
}/////////////////////////////////////////实现的卷积方法，步长为1且无padding
double Max(double x, double y)
{
	if (x >= y)
		
	{
		return x;
	}
	else
	{
		return y;
	}
}////////////////////////最大值函数，用来计算maxpooling
/// </summary>
void meanpooling(Mat input, Mat& output, int pool_size)
{
	int in_row = input._row_num;
	int in_col = input._col_num;
	int out_row = output._row_num;
	int out_col = output._col_num;
	double sum;
	for (int i = 0; i < out_row; i++)
	{
		for (int j = 0; j < out_col; j++)
		{
			sum = 0;
			for (int u =i*pool_size; u <(i+1)*pool_size; u++)
			{
				for (int v = j*pool_size; v <(j+1)*pool_size; v++)
				{
					sum += input.ptr[u][v];
				}
			}
			output.ptr[i][j] = sum / double(pool_size * pool_size);
		}
	}
}////////////////////////////////////////平均池化函数
void maxpooling(Mat input, Mat& output,Mat &max_position, int pool_size)
{
	int in_row = input._row_num;
	int in_col = input._col_num;
	int out_row = output._row_num;
	int out_col = output._col_num;
	double max;
	for (int i = 0; i < out_row; i++)
	{
		for (int j = 0; j < out_col; j++)
		{
			max = 0;
			for (int u = i * pool_size; u < (i + 1) * pool_size; u++)
			{
				for (int v = j * pool_size; v < (j + 1) * pool_size; v++)
				{
					if (u == i * pool_size && v == j * pool_size)
					{
						max = input.ptr[u][v];
						max_position.ptr[i][j] = u * in_col + v;
					}
					else
					{
						max = Max(max, input.ptr[u][v]);
						max_position.ptr[i][j] = u * in_col + v;
					}
				}
			}
			output.ptr[i][j] = max;
		}
	}
}//////////////////////////////////////////////最大池化函数
double Relu(double w, double b)
{
	if (w + b > 0)
	{
		return w + b;
	}
	else
	{
		return 0;
	}
}////////////////////////////////////////激活函数
void softmax(Outlayer& O1)
{
	double sum = 0.0;
	for (int i = 0; i < O1._out_num; i++)
	{
		O1._out_ac.ptr[0][i] = exp(O1._b.ptr[0][i] + O1._in_ac.ptr[0][i]);
		sum += O1._out_ac.ptr[0][i];
	}
	for (int i = 0; i < O1._out_num; i++)
	{
		O1._out_ac.ptr[0][i] = O1._out_ac.ptr[0][i] / sum;
	}
}//////////////////////////////softmax函数，用来将dense层的输出均一化
void cov_layer_fw(vector<Mat> input, Convolutional_layer& C1)///这里的input是不是一个vector取决于一次性进去多少个
{
	for (int i = 0; i < C1._outChannels; i++)
	{
		for (int j = 0; j < C1._inChannels; j++)
		{
			Mat cor = Convolution(input[j], C1._cov_weight[j][i]);
			/*			for (int e = 0; e < 24; e++)
			  {
			  for (int t = 0; t < 24; t++)
			  {
			  cout << cor.ptr[e][t] << " ";
			  }
			  cout << "\n";
			  }*//////////////////////////////////////////////这个地方有问题
			C1._in_ac[i] = Addition(C1._in_ac[i],cor);
		}
		//cout << i<<"\n";
		//for (int e = 0; e < C1._in_ac[i]._row_num; e++)
		//{
		//	for (int t = 0; t < C1._in_ac[i]._col_num; t++)
		//	{
		//		cout << C1._in_ac[i].ptr[e][t] << " ";
		//	}
		//	cout << "\n";
		//}
		int out_row = C1._in_ac[i]._row_num;
		int out_col = C1._in_ac[i]._col_num;
		for (int u = 0; u < out_row; u++)
		{
			for (int v=0;v<out_col;v++)
			{
				C1._out_ac[i].ptr[u][v] = Relu(C1._in_ac[i].ptr[u][v], C1._b.ptr[0][i]);
			}
		}
	}
	
}//////////////////////////////卷积层进行前向传播的函数
void pool_layer_fw(vector<Mat> input, Poollayer& P1, int pool_type=1)
{
	if (pool_type == 0)
	{
		for (int i = 0; i < P1._outChannels; i++)
		{
			meanpooling(input[i], P1._out[i], P1._pool_size);
		}
	}
	else
	{
		for (int i = 0; i < P1._outChannels; i++)
		{
			maxpooling(input[i], P1._out[i], P1.max_position[i], P1._pool_size);
		}
	}
}/////////////////////////池化层进行前向传播的函数
void weight_mult(Mat input, Mat weight, Mat& output)

{
	int sum;
	for (int i = 0; i < output._col_num; i++)
	{
		sum = 0;
		for (int j = 0; j < weight._col_num; j++)
		{
			sum += input.ptr[0][j] * weight.ptr[i][j];
		}
		output.ptr[0][i] = sum;
	}
}///////////////////////////////////////////用于dense层的权重计算函数，对应公式为w*x+b
void out_layer_fw(vector<Mat>input, Outlayer& O1)
{
	Mat temp(1, O1._in_num);
	int row = input[0]._row_num;
	int col = input[0]._col_num;
	int Channels_num = input.size();
	for (int i = 0; i < Channels_num; i++)
	{
		for (int j = 0; j < row; j++)
		{
			for (int k = 0; k < col; k++)
			{
				temp.ptr[0][i * row * col + j * col + k] = input[i].ptr[j][k];
			}
		}
	}
	weight_mult(temp, O1._weight, O1._in_ac);
	softmax(O1);
}///////////////////////////////////输出层的前向传播函数

void cnn_fw(CNN& cnn, Mat inputData)
{
	//C1
	//5*5卷积核
	//输入28*28矩阵
	//输出(28-25+1)*(28-25+1) = 24*24矩阵
	vector<Mat> input_tmp;
	input_tmp.push_back(inputData);
	cout << "input";
	//for (int i = 0; i < inputData._row_num; i++)
	//{
	//	for (int j = 0; j < inputData._col_num; j++)
	//	{
	//		cout << inputData.ptr[i][j]<<" ";
	//	}
	//	cout << "\n";
	//}
	cov_layer_fw(input_tmp, cnn.C1);//这里的输入还要按照后面讲的来说。
	//cout << "CNN c1.inac\n";//////////////////////////////////////////////grad这个全是0，反向传播有问题
	//for (int i = 0; i <cnn.C1._grad[0]._row_num; i++)
	//{
	//	for (int j = 0; j < cnn.C1._grad[0]._col_num; j++)
	//	{
	//		cout << cnn.C1._grad[0].ptr[i][j] << " ";
	//	}
	//	cout << "\n";
	//}
	//cout << "out_ac\n";
	//for (int i = 0; i < cnn.C1._out_ac[0]._row_num; i++)
	//{
	//	for (int j = 0; j < cnn.C1._out_ac[0]._col_num; j++)
	//	{
	//		cout << cnn.C1._out_ac[0].ptr[i][j] << " ";
	//	}
	//	cout << "\n";
	//}
	//cout << "cov weight\n";
	//for (int i = 0; i < cnn.C1._cov_weight[0][0]._row_num; i++)
	//{
	//	for (int j = 0; j < cnn.C1._cov_weight[0][0]._col_num; j++)
	//	{
	//		cout << cnn.C1._cov_weight[0][0].ptr[i][j] << " ";
	//	}
	//	cout << "\n";
	//}
	//P2
	//24*24-->12*12
	pool_layer_fw(cnn.C1._out_ac,cnn.P2,1);
	//cout << "P2 -out\n";
	//for (int i = 0; i < cnn.P2._out[0]._row_num; i++)
	//{
	//	for (int j = 0; j < cnn.P2._out[0]._col_num; j++)
	//	{
	//		cout << cnn.P2._out[0].ptr[i][j]<<" ";
	//	}
	//	cout << "\n";
	//}
	
	//C3
	//12*12-->8*8
	cov_layer_fw(cnn.P2._out, cnn.C3);//////////上面的到这里也有显现
	//cout << "CNN c3.inac\n";
	//for (int i = 0; i < cnn.C3._in_ac[0]._row_num; i++)
	//{
	//	for (int j = 0; j < cnn.C3._in_ac[0]._col_num; j++)
	//	{
	//		cout << cnn.C3._in_ac[0].ptr[i][j] << " ";
	//	}
	//	cout << "\n";
	//}
	//cout << "out_ac\n";
	//for (int i = 0; i < cnn.C3._out_ac[0]._row_num; i++)
	//{
	//	for (int j = 0; j < cnn.C3._out_ac[0]._col_num; j++)
	//	{
	//		cout << cnn.C3._out_ac[0].ptr[i][j] << " ";
	//	}
	//	cout << "\n";
	//}
	//cout << "cov weight\n";
	//for (int i = 0; i < cnn.C3._cov_weight[0][0]._row_num; i++)
	//{
	//	for (int j = 0; j < cnn.C3._cov_weight[0][0]._col_num; j++)
	//	{
	//		cout << cnn.C3._cov_weight[0][0].ptr[i][j] << " ";
	//	}
	//	cout << "\n";
	//}
	
	//P4
	//8*8-->4*4
	pool_layer_fw(cnn.C3._out_ac, cnn.P4,1);
	
	//cout << "P4 -out\n";
	//for (int i = 0; i < cnn.P4._out[0]._row_num; i++)
	//{
	//	for (int j = 0; j < cnn.P4._out[0]._col_num; j++)
	//	{
	//		cout << cnn.P4._out[0].ptr[i][j] << " ";
	//	}
	//	cout << "\n";
	//}
	
	
	
	//O5
	//12*4*4-->192-->1*10
	out_layer_fw(cnn.P4._out, cnn.O5);
	
	//for (int i = 0; i < cnn.O5._grad._row_num; i++)
	//{
	//	for (int j = 0; j < cnn.O5._grad._col_num; j++)
	//	{
	//		cout << cnn.O5._grad.ptr[i][j] << " ";
	//	}
	//	cout << "\n";
	//}
	//cout << "O5-inac\n";
	//for (int i = 0; i < cnn.O5._in_ac._row_num; i++)
	//{
	//	for (int j = 0; j < cnn.O5._in_ac._col_num; j++)
	//	{
	//		cout << cnn.O5._in_ac.ptr[i][j]<<" ";
	//	}
	//	cout << "\n";
	//}
	//cout << "outac\n";
	//for (int i = 0; i < cnn.O5._out_ac._row_num; i++)
	//{
	//	for (int j = 0; j < cnn.O5._out_ac._col_num; j++)
	//	{
	//		cout << cnn.O5._out_ac.ptr[i][j] << " ";
	//	}
	//	cout << "\n";
	//}
}///////////////////////////////////////////////////前向传播的函数，用于调用对应层的前向传播函数
void softmax_bp(Mat output, Mat& e, Outlayer &O1)
{
	for (int i = 0; i < O1._out_num; i++)
	{
		e.ptr[0][i] = O1._out_ac.ptr[0][i] - output.ptr[0][i];
	}
	for (int i = 0; i < O1._out_num; i++)
	{
		O1._grad.ptr[0][i] = e.ptr[0][i];
	}
	//cout << "softmax ok";
}///////////////////////////////softmax层的反向传播函数
void dense_pool_bp(Outlayer O1, Poollayer& P1)
{
	int row = P1._inhigh / P1._pool_size;
	int col = P1._inwide / P1._pool_size;
	for (int i = 0; i < P1._outChannels; i++)
	{
		for (int j = 0; j < row; j++)
			
		{
			for (int k = 0; k < col; k++)
			{
				for (int u = 0; u < O1._out_num; u++)
				{
					P1._grad[i].ptr[j][k] = P1._grad[i].ptr[j][k] + O1._grad.ptr[0][u] * O1._weight.ptr[u][i * row * col + j * col + k];
				}
			}
		}
	}
	//cout << "dense_pool ok";
}////////////////////////全连接层的反向连接函数
Mat extend(Mat input, int cov_size)
{
	Mat temp(input._row_num + 2 * (cov_size - 1), input._col_num + 2 * (cov_size - 1));
	for (int i = cov_size - 1; i < input._col_num + cov_size - 1; i++)
	{
		for (int j = cov_size - 1; j < input._col_num + cov_size - 1; j++)
		{
			temp.ptr[i][j] = input.ptr[i - cov_size + 1][j - cov_size + 1];
		}
	}
	return temp;
}///////////////////////////拓展函数，用于池化层的反向传播
Mat turn_over(Mat input)
{
	Mat temp(input._row_num, input._col_num);
	for (int i = 0; i < input._row_num; i++)
	{
		for (int j = 0; j < input._col_num; j++)
		{
			temp.ptr[input._row_num - i-1][input._col_num - j-1] = input.ptr[i][j];
		}
	}
	return temp;
}//////////////////////////////反转函数，将输入矩阵反转180°，用于后续的求导过程。
Mat cov_grad(Mat input, Mat cov_weight)
{
	Mat temp = turn_over(cov_weight);
	Mat ext = extend(input, cov_weight._row_num);
	Mat ans = Convolution(ext, temp);
	return ans;
}///////////////////对于卷积层的求导函数

void cov_pool_bp(Covlayer C1, Poollayer& P1)
{
	for (int i = 0; i < P1._outChannels; i++)
	{
		for (int j = 0; j < P1._inChannels; j++)
		{
			Mat res = cov_grad(C1._grad[j], C1._cov_weight[i][j]);////////////////////////////////打个标记
			P1._grad[i] = Addition(P1._grad[i] , res);
		}
	}
}///////////卷积层的反向传播函数
void update_dense_para(vector<Mat> input, CNNOpts opt, Outlayer O1)
{
	int row = input[0]._row_num;
	int col = input[0]._col_num;
	Mat O1in(1, row * col * input.size());
	for (int i = 0; i < input.size(); i++)
	{
		for (int j = 0; j < row; j++)
		{
			for (int k = 0; k < col; k++)
			{
				O1in.ptr[0][i * row * col + j * col + k] = input[i].ptr[j][k];
			}
		}
	}
	for (int j = 0; j < O1._out_num; j++)  //10通道
	{
		for (int i = 0; i < O1._in_num; i++)  //192通道
		{
			//w = w - α。dE/dw
			O1._weight.ptr[j][i] = O1._weight.ptr[j][i] - opt._alpha * O1._grad.ptr[0][j] * O1in.ptr[0][i];
		}
		//b = b - α。dE/db
		O1._b.ptr[0][j] = O1._b.ptr[0][j] - opt._alpha * O1._grad.ptr[0][j];
	}
}
//////////////////////全连接层的参数更行函数

void update_cov_para(vector<Mat> input, CNNOpts opts, Covlayer& C)
{
	for (int i = 0; i < C._outChannels; i++)   //6通道
	{
		for (int j = 0; j < C._inChannels; j++)   //1通道
		{
			Mat temp = Convolution(input[j],C._grad[i]); 
			temp = Multiple( temp ,(-opts._alpha));   //矩阵乘以系数-α.dE/dk
			C._cov_weight[j][i] = Addition(C._cov_weight[j][i], temp);   //计算k = k - α.dE/dk
		}
		//cout << "sum wenti";
		double d_sum = Sum(C._grad[i]);   //计算sum(dC3)，这里有6个24*24的d，6个偏置b，一个偏置b对应一个24*24矩阵d的所有元素和////////////////////这里有部分不同需要考虑。
		C._b.ptr[0][i] = C._b.ptr[0][i] - opts._alpha * d_sum;  //计算b = b - α.dE/db
	}
}


Mat mean_UpSample(Mat& input, int pool_size)
{
	int row = input._row_num;
	int col = input._col_num;
	Mat temp(row * pool_size, col * pool_size);
	
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			double num = input.ptr[i][j] / (pool_size * pool_size);
			for (int k = 0; k < pool_size; k++)
			{
				for (int u = 0; u < pool_size; u++)
				{
					temp.ptr[i * pool_size + k][j * pool_size + u] = num;;
				}
			}
		}
	}
	return temp;
}
Mat maxUpSample(Mat& input, Mat max_position, int pool_size)
{
	int row = input._row_num;
	int col = input._col_num;
	int out_row = row * pool_size;
	int out_col = col * pool_size;
	Mat temp(out_row, out_col);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			int max_r = int(max_position.ptr[i][j]) / out_col;
			int max_c = int(max_position.ptr[i][j]) % out_col;
			temp.ptr[max_r][max_c] = input.ptr[i][j];
		}
	}
	return temp;
}
double UnRelu(double x)
{
	if (x > 0)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}//////////////////////////Relu函数的反函数，如果该值为大于0则选择为1否则则为0
void pool_cov_bp(Poollayer P1, Covlayer& C1)
{
	for (int i = 0; i < C1._outChannels; i++)
	{
		Mat temp;
		if (P1._pool_type == 0)
		{
			temp = mean_UpSample(P1._grad[i], P1._pool_size);
		}
		else
		{
			temp = maxUpSample(P1._grad[i], P1.max_position[i], P1._pool_size);
		}
		
		for (int j = 0; j < P1._inhigh; j++)
		{
			for (int k = 0; k < P1._inwide; k++)
			{
				C1._grad[i].ptr[j][k] = temp.ptr[j][k] * UnRelu(C1._out_ac[i].ptr[j][k]);
			}
		}
	}
}

//////////////////////////////////////////////所有参数的更新以及参数的清零后续再做吧
//outputData为标签
void cnnbp(CNN& cnn, Mat outputData)
{
	softmax_bp(outputData, cnn.e, cnn.O5);
	//cout << "1 ok";
	dense_pool_bp(cnn.O5, cnn.P4);
	//for (int i = 0; i < cnn.P4._grad[0]._row_num; i++)
	//{
	//	for (int j = 0; j < cnn.P4._grad[0]._col_num; j++)
	//	{
	//		cout << cnn.P4._grad[0].ptr[i][j] << " ";
	//	}
	//	cout << "\n";
	//}
	//cout << "2 ok";
	pool_cov_bp(cnn.P4, cnn.C3);
	
	//cout << "3 ok";
	cov_pool_bp(cnn.C3,cnn.P2);
	//	for (int i = 0; i < cnn.P2._grad[0]._row_num; i++)
	//{
	//	for (int j = 0; j < cnn.P2._grad[0]._col_num; j++)
	//	{
	//		cout << cnn.P2._grad[0].ptr[i][j] << " ";
	//	}
	//	cout << "\n";
	//}
	//cout << "4 ok";
	pool_cov_bp(cnn.P2, cnn.C1);
	//for (int i = 0; i < cnn.C1._grad[0]._row_num; i++)
	//{
	//	for (int j = 0; j < cnn.C1._grad[0]._col_num; j++)
	//	{
	//		cout << cnn.C1._grad[0].ptr[i][j] << " ";
	//	}
	//	cout << "\n";
	//}
	//cout << "5 ok";
}///////////////////////////////////////////////////超级拼装，用来把所有反向传播合在一起的

//将int数据中的4个字节数据按相反顺序重新排列，重组成一个int数据




void cnnapplygrads(CNN& cnn, CNNOpts opts, Mat inputData) // 更新权重
{
	vector<Mat> input_tmp;
	input_tmp.push_back(inputData);
	
	
	update_cov_para(input_tmp, opts, cnn.C1);
	//cout << "1 ok";
	
	update_cov_para(cnn.P2._out, opts, cnn.C3);
	//cout << "2 ok";
	
	update_dense_para(cnn.P4._out, opts, cnn.O5);
	//cout << "3 ok";
}
void clear_cov_para(Covlayer& C1)
{
	int row = C1._grad[0]._row_num;
	int col = C1._grad[0]._col_num;
	for (int i = 0; i < C1._outChannels; i++)
	{
		for (int j = 0; j < row; j++)
		{
			for (int k = 0; k < col; k++)
			{
				C1._grad[i].ptr[j][k] = 0.0;
				C1._in_ac[i].ptr[j][k] = 0.0;
				C1._out_ac[i].ptr[j][k] = 0.0;
			}
		}
	}
}
void clear_pool_para(Poollayer& P1)
{
	int row = P1._grad[0]._row_num;
	int col = P1._grad[0]._col_num;
	for (int i = 0; i < P1._outChannels; i++)
	{
		for (int j = 0; j < row; j++)
		{
			for (int k = 0; k < col; k++)
			{
				P1._grad[i].ptr[j][k] = 0.0;
				P1._out[i].ptr[j][k] = 0.0;
			}
		}
	}
}
void clear_out_para(Outlayer& O1)
{
	for (int i = 0; i < O1._out_num; i++)
	{
		O1._grad.ptr[0][i] = 0.0;
		O1._in_ac.ptr[0][i] = 0.0;
		O1._out_ac.ptr[0][i] = 0.0;
	}
}
void clear(CNN& cnn)
{
	clear_cov_para(cnn.C1);
	//cout << "1ok";
	clear_pool_para(cnn.P2);
	//cout << "2 ok";
	clear_cov_para(cnn.C3);
	//cout << "3 ok";
	clear_pool_para(cnn.P4);
	//cout << "4 ok";
	clear_out_para(cnn.O5);
	//cout << "5 ok";
}////////////////////////清除参数函数，避免梯度累加


void cnntrain(CNN& cnn, vector<Mat> inputData, vector<Mat> outputData, CNNOpts opts, int trainNum)
{
	// 学习训练误差曲线，记录交叉熵误差函数的值
	cnn.L = Mat(1, trainNum).clone();
	double lr = opts._alpha;
	for (int e = 0; e < opts._numepochs; e++)   //opts.numepochs表示需要训练次数
	{
		for (int n = 0; n < trainNum; n++)   //trainNum表示由多少张图片，训练完这些图片相当于完成一次训练
		{
			//学习率递减0.03~0.001
			opts._alpha = lr - 0.00029 * n / (trainNum - 1);
			
			cnn_fw(cnn, inputData[n]);   // 前向传播 
			//cout << "前向转播可行";
			cnnbp(cnn, outputData[n]); // 后向传播
			//cout << "反向传播可行";
			cnnapplygrads(cnn, opts, inputData[n]); // 更新参数
			//cout << "更新参数可行";
			// 计算交叉熵误差函数的值
			float l = 0.0;
			for (int i = 0; i < cnn.O5._out_num; i++)
			{
				l = l - outputData[n].ptr[0][i] * log(cnn.O5._out_ac.ptr[0][i]);
				//cout << cnn.O5._out_ac.ptr[0][i] << " ";
			}
			//cout << "\n";
			cnn.L.ptr[0][n] = l;
			
			clear(cnn);   //清零参数
			
			
			printf("n=%d, f=%f, α=%f\n", n, cnn.L.ptr[0][n], opts._alpha);
		}
	}
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 0xff;
	ch2 = (i >> 8) & 0xff;
	ch3 = (i >> 16) & 0xff;
	ch4 = (i >> 24) & 0xff;
	
	
	return ((int)(ch1 << 24)) + ((int)(ch2 << 16)) + ((int)(ch3 << 8)) + (int)ch4;
}
void Trans(Mat& temp, unsigned char* temp_ptr)
{
	int row = temp._row_num;
	int col = temp._col_num;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			temp.ptr[i][j] = double(int(temp_ptr[i*col+j]));
		}
	}
}
void Division(Mat& temp, double x)
{
	int row = temp._row_num;
	int col = temp._col_num;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			temp.ptr[i][j] = temp.ptr[i][j] / x;
		}
	}
}
//1行n列的向量
int Prediction(Mat output)  //返回向量最大数的序号
{
	int col = output._col_num;
	float maxnum = -1.0;
	int maxIndex = 0;
	for (int i = 0; i <col; i++)
	{
		if (maxnum < output.ptr[0][i])
		{
			maxnum = output.ptr[0][i];
			maxIndex = i;
		}
	}
	return maxIndex;
}


//测试函数
double cnntest(CNN cnn, vector<Mat> input, vector<Mat> output)
{
	int incorrectnum = 0;  //错误预测的数目
	for (int i = 0; i < input.size(); i++)  //inputData.size()为测试图像的总数
	{
		cnn_fw(cnn, input[i]);   //前向传播
		//检查神经网络输出的最大概率的序号是否等于标签中1值的序号，如果等于则表示分类成功
		if (Prediction(cnn.O5._out_ac) != Prediction(output[i]))
		{
			incorrectnum++;
			printf("i = %d, 识别失败\n", i);
		}
		else
		{
			printf("i = %d, 识别成功\n", i);
		}
		clear(cnn);
	}
	printf("incorrectnum=%d\n", incorrectnum);
	printf("inputData.size()=%d\n", input.size());
	return (double)incorrectnum / (double)input.size();
}
vector<Mat> read_Img_to_Mat(const char* filename)
{
	FILE* fp = NULL;
	fp = fopen(filename, "rb");
	if (fp == NULL)
		printf("open file failed\n");
	assert(fp);
	
	
	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;
	
	fread(&magic_number, sizeof(int), 1, fp);   //从文件中读取sizeof(int) 个字符到 &magic_number  
	magic_number = ReverseInt(magic_number);
	
	
	fread(&number_of_images, sizeof(int), 1, fp);   //获取训练或测试image的个数number_of_images 
	number_of_images = ReverseInt(number_of_images);
	
	fread(&n_rows, sizeof(int), 1, fp);   //获取训练或测试图像的高度Heigh  
	n_rows = ReverseInt(n_rows);
	
	fread(&n_cols, sizeof(int), 1, fp);   //获取训练或测试图像的宽度Width  
	n_cols = ReverseInt(n_cols);
	
	
	
	//获取第i幅图像，保存到vec中 
	int i, r, c;
	
	
	int img_size = n_rows * n_cols;
	vector<Mat> img_list;
	//char** temp_ptr = new char* [n_rows];
	//for (int i = 0; i < n_cols; i++)
	//{
	//	temp_ptr[i] = new char[n_cols];
	//}
	unsigned char* temp_ptr = new unsigned char[n_cols * n_rows];
	for (i = 0; i < number_of_images; ++i)
	{
		Mat tmp(n_rows, n_cols);
		
		fread(temp_ptr, sizeof(unsigned char), img_size, fp);  //读取一张图像
		Trans(tmp, temp_ptr);   //将图像转换为float数据
		Division(tmp, 255.0);   //将数据转换成0~1的数据
		img_list.push_back(tmp.clone());
	}
	//for (int i = 0; i < n_cols; ++i) {
	//	delete[] temp_ptr[i];
	//}
	delete[] temp_ptr;
	
	fclose(fp);
	return img_list;
}/////////////////////////这里在读取之后用的是int类型进行接受的，需要加一个东西来改变//改变完毕

vector<Mat> read_Lable_to_Mat(const char* filename)
{
	FILE* fp = NULL;
	fp = fopen(filename, "rb");
	if (fp == NULL)
		printf("open file failed\n");
	assert(fp);
	
	
	int magic_number = 0;
	int number_of_labels = 0;
	int label_long = 10;
	
	
	
	fread(&magic_number, sizeof(int), 1, fp);   //从文件中读取sizeof(magic_number) 个字符到 &magic_number  
	magic_number = ReverseInt(magic_number);
	
	
	fread(&number_of_labels, sizeof(int), 1, fp);   //获取训练或测试image的个数number_of_images 
	number_of_labels = ReverseInt(number_of_labels);
	
	
	int i, l;
	
	
	vector<Mat> label_list;
	
	for (i = 0; i < number_of_labels; ++i)
	{
		
		Mat tmp = Mat::zero(1, label_long);
		unsigned char temp = 0;
		fread(&temp, sizeof(unsigned char), 1, fp);
		tmp.ptr[0][(int)temp] = 1.0;  //将0~9的数字转换成one-hot码
		label_list.push_back(tmp.clone());
	}
	
	
	fclose(fp);
	return label_list;
}
void save_cnn_para(const char* filename,CNN& cnn)
{
	FILE* fp = NULL;
	fp = fopen(filename, "wb");
	if (fp == NULL)
		printf("open file failed\n");
	assert(fp);
	for (int i = 0; i < cnn.C1._inChannels; i++)
	{
		for (int j = 0; j < cnn.C1._outChannels; j++)
		{
			for (int u = 0; u < cnn.C1._cov_size; u++)
			{
				for (int v = 0; v < cnn.C1._cov_size; v++)
				{
					fwrite(&(cnn.C1._cov_weight[i][j].ptr[u][v]), sizeof(double), 1, fp);
				}
			}
			fwrite(&(cnn.C1._b.ptr[0][j]), sizeof(double), 1, fp);
		}
	}
	for (int i = 0; i < cnn.C3._inChannels; i++)
	{
		for (int j = 0; j < cnn.C3._outChannels; j++)
		{
			for (int u = 0; u < cnn.C3._cov_size; u++)
			{
				for (int v = 0; v < cnn.C3._cov_size; v++)
				{
					fwrite(&(cnn.C3._cov_weight[i][j].ptr[u][v]), sizeof(double), 1, fp);
				}
			}
			fwrite(&(cnn.C1._b.ptr[0][j]), sizeof(double), 1, fp);
		}
	}
	for (int i = 0; i < cnn.O5._out_num; i++)
	{
		for (int j = 0; j < cnn.O5._in_num; j++)
			
		{
			fwrite(&(cnn.O5._weight.ptr[i][j]), sizeof(double), 1, fp);
		}
		fwrite(&(cnn.O5._b.ptr[0][i]), sizeof(double), 1, fp);
	}
	fclose(fp);
}
void config_cnn_para(const char* filename, CNN& cnn)
{
	FILE* fp = NULL;
	fp = fopen(filename, "rb");
	if (fp == NULL)
		printf("open file failed\n");
	assert(fp);
	double para = 0.0;
	for (int i = 0; i < cnn.C1._inChannels; i++)
	{
		for (int j = 0; j < cnn.C1._outChannels; j++)
		{
			for (int u = 0; u < cnn.C1._cov_size; u++)
			{
				for (int v = 0; v < cnn.C1._cov_size; v++)
				{
					fread(&para, sizeof(double), 1, fp);
					cnn.C1._cov_weight[i][j].ptr[u][v] = para;
				}
			}
			fread(&para, sizeof(double), 1, fp);
			cnn.C1._b.ptr[0][j] = para;
		}
	}
	for (int i = 0; i < cnn.C3._inChannels; i++)
	{
		for (int j = 0; j < cnn.C3._outChannels; j++)
		{
			for (int u = 0; u < cnn.C3._cov_size; u++)
			{
				for (int v = 0; v < cnn.C3._cov_size; v++)
				{
					fread(&para, sizeof(double), 1, fp);
					cnn.C3._cov_weight[i][j].ptr[u][v] = para;
				}
			}
			fread(&para, sizeof(double), 1, fp);
			cnn.C3._b.ptr[0][j] = para;
		}
	}
	for (int i = 0; i < cnn.O5._out_num; i++)
	{
		for (int j = 0; j < cnn.O5._in_num; j++)
			
		{
			fread(&para, sizeof(double), 1, fp);
			cnn.O5._weight.ptr[i][j] = para;
		}
		
		fread(&para, sizeof(double), 1, fp);
		cnn.O5._b.ptr[0][i] = para;
	}
}
//int main()
//{

//	Mat m1(28, 28);
//	Mat m2(5, 5);
//	for (int i = 0; i < 28; i++)
//	{
//		for (int j = 0; j < 28;j++)
//		{
//			m1.ptr[i][j] = i + j;
//		}
//	}
//	for (int i = 0; i < 5; i++)
//	{
//		for (int j = 0; j < 5; j++)
//		{
//			m2.ptr[i][j] = i + j;
//		}
//	}
//	Mat m3 = Convolution(m1, m2);
//	//for (int i = 0; i<24; i++)
//	//{
//	//	for (int j = 0; j < 24; j++)
//	//	{
//	//		cout << m3.ptr[i][j] << " ";
//	//	}
//	//	cout << "\n";
//	//}

//}
//	int main()
//	{
//		CNN cnn;
//		cnnsetup(cnn, 28, 28, 10);
//		for (int i = 0; i < 5; i++)
//		{
//			for (int j = 0; j < 5; j++)
//			{
//				cout << cnn.C1._cov_weight[0][1].ptr[i][j]<<" ";
//			}
//			cout << "\n";
//		}
//		save_cnn_para("C:\\Users\\yuanyi\\Desktop\\para.txt", cnn);
//		config_cnn_para("C:\\Users\\yuanyi\\Desktop\\para.txt", cnn);
//		for (int i = 0; i < 5; i++)
//		{
//			for (int j = 0; j < 5; j++)
//			{
//				cout << cnn.C1._cov_weight[0][1].ptr[i][j] << " ";
//			}
//			cout << "\n";
//		}
//}
int main()
{
	int cmd;
	cout << "欢迎使用CNN卷积神经网络，本模型由卷积层->Relu->最大池化层->卷积层->Relu->卷积层->dense->softmax->输出层构成\n";
	vector<Mat> traindata_list;
	vector<Mat> traindata_label;
	vector<Mat> testdata_list;
	vector<Mat> testdata_label;
	cout << "读取训练数据标签\n";
	//读取训练数据标签
	traindata_label = read_Lable_to_Mat("C:\\Users\\yuanyi\\Desktop\\MNIST\\train-labels.idx1-ubyte");
	cout << "读取训练数据标签完成\n";
	cout << "读取训练数据\n";
	//读取训练数据
	traindata_list = read_Img_to_Mat("C:\\Users\\yuanyi\\Desktop\\MNIST\\train-images.idx3-ubyte");
	cout << "训练数据读取完成\n";
	cout << "读取测试数据标签\n";
	//读取测试数据标签
	testdata_label = read_Lable_to_Mat("C:\\Users\\yuanyi\\Desktop\\MNIST\\t10k-labels.idx1-ubyte");
	cout << "读取测试数据标签完成\n";
	cout << "读取测试数据\n";
	//读取测试数据
	testdata_list = read_Img_to_Mat("C:\\Users\\yuanyi\\Desktop\\MNIST\\t10k-images.idx3-ubyte");
	cout << "读取测试数据完成\n";
	//for (int i = 0; i < traindata_label.size(); i++)
	//{
	//	for (int j = 0; j < 10; j++)
	
	//	{
	//		cout << traindata_label[i].ptr[0][j]<<" ";
	//	}
	//	cout << "\n";
	//}
	int train_num = traindata_list.size();
	int test_num = testdata_list.size();
	int outSize = testdata_label[0]._col_num;
	int a, b;
	
	int row = traindata_list[0]._row_num;
	int col = traindata_list[0]._row_num;
	int signal = 0;
	
	CNNOpts opts;
	opts._numepochs = 1;
	opts._alpha = 0.0003;   //学习率初始值
	int trainNum = 60000;
	
	
	CNN cnn;
	cout << "      以下为对应使用菜单      \n";
	cout << "      0:采用mnist数据集中的训练集进行训练      \n";
	cout << "      1:对于mnist中的测试集进行预测并计算准确率      \n";
	cout << "      2:加载菜单      \n";
	cout << "      3:加载参数      \n";
	cout << "      4：保存训练后模型参数      \n";
	cout<<"      press ^Z to exit      \n";
	a = 0; 
	b = 0;
	float success = 0;
	while (cin >> cmd)
	{
		if (cmd == 0)
		{
			
			//for (int i = 0; i < 24; i++)
			//{
			//	for (int j = 0; j < 24; j++)
			//	{
			//		cout <<m1.ptr[i][j] << " ";
			//	}
			//	cout << "\n";
			//}
			if (a != 0)
			{
				cout << "模型已被训练过，无法使用同一数据集二次训练\n";
			}
			else
			{
				if (signal == 0)
				{
					cout << "模型正在初始化\n";
					cnnsetup(cnn, row, col, outSize);   //cnn初始化
					cout << "模型初始化完成\n";
					cout << "模型开始训练\n";
					cnntrain(cnn, traindata_list, traindata_label, opts, 60000);  //训练
					cout << "模型训练完成\n";
					a++;
					signal++;
				}
				else {
					cout << "模型已通过已有文件加载，请勿再次训练";
				}
			}
			
		}
		else if (cmd == 1)
		{
			if (b == 0)
			{
				cout << "模型开始测试";
				success = cnntest(cnn, testdata_list, testdata_label);   //分类
				cout << "模型测试结束";
				b++;
				printf("success=%f\n", 1 - success);
			}
			else
			{
				printf("success=%f\n", 1 - success);
			}
			
		}
		else if (cmd == 2)
		{
			cout << "      以下为对应使用菜单      \n";
			cout << "      0:采用mnist数据集中的训练集进行训练      \n";
			cout << "      1:对于mnist中的测试集进行预测并计算准确率      \n";
			cout << "      2:加载菜单      \n";
			cout << "      3:加载参数      \n";
			cout << "      4：保存训练后模型参数      \n";
			cout << "      press ^Z to exit      \n";
		}
		else if (cmd == 3)
		{
			if (signal == 0)
			{
				cout << "模型正在初始化\n";
				cnnsetup(cnn, row, col, outSize);   //cnn初始化
				cout << "模型初始化完成\n";
				cout << "模型正在加载参数\n";
				config_cnn_para("C:\\Users\\yuanyi\\Desktop\\para.txt", cnn);
				cout << "模型加载参数完成\n";
				signal++;
			}
			else
			{
				cout << "模型已被训练，请勿随机加载文件参数\n";
			}
		}
		else if (cmd == 4)
		{
			if (a == 0&&signal==0)
			{
				cout << "模型尚未训练或加载参数，无法进行参数保存\n";
			}
			else
			{
				cout << "开始保存模型参数\n";
				save_cnn_para("C:\\Users\\yuanyi\\Desktop\\para.txt", cnn);
				cout << "模型参数保存完毕\n";
			}
		}
	}
	return 0;
}//需要检查 vector//休息一下，后面在做


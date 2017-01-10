/*
* description: 
*        基于Per-Coordinate FTRL-Proximal 算法的logistic回归的实现.算法的细节参见下文:
*        McMahan H B, Holt G, Sculley D, et al. Ad click prediction: a view from the trenches[C]
*        Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2013: 1222-1230.
*/
#ifndef FTRL_PROXIMAL_HPP
#define FTRL_PROXIMAL_HPP

#include <cmath>
#include <map>
#include <iostream>
#include <fstream> 
using namespace std;

namespace FTRL
{
    long max_feature_num = 1048576;//最大特征个数,我们的服务器可以轻松支持500W个特征
    typedef map<long,double> m_FeatureItems;// 稀疏特征表示,key为特征索引,value为特征值,本程序总是假设特征的量纲都是[-1,1],所以程序中没有做任何数据标准化

   /////符号函数
    inline int sgn(double z)
    {
        int sign = 0;
        if(z>0)
        {    
            sign = 1;
        }
        if(z<0)
        {
            sign = -1;
        }
        return sign;
    }
    //////计算logloss
    inline double logloss(double p,double y)
    {
        if(p<10e-15)/////预测概率值需要大于10e-15
        {
            p = 10e-15;
        }
        if(p>1-10e-15)/////预测概率需要小于1-10e-15
        {
            p=1-10e-15; 
        }
        return  -y*log(p)-(1-y)*log(1-p);
    }

    ////////模型训练类,
    ///////注意本训练类只能在给定参数和特征的情况下使用,而且不对样本进行概率采样(采样应当在上游完成).如果要对比增减特征或者不同的参数设置后模型的效果，
    ///////那么目前采用流拷贝的方式(其实只有在下游模型不在同一台机器的时候才需要留拷贝，在同一台机器的话需要server自己解析协议后后拷贝分发)
    ///////首先需要上游进行特征选取，下游将管理者多个本类对象，根据到来的不同数据和参数指定扔给不同的对象。
    class CFtrlAlgorithm
    {
    public:
        CFtrlAlgorithm(long _d=max_feature_num,double _lambda1=1,double _lambda2=1,double _alpha=0.1,int _calcloss_gap=100,double _beta=1)
            :d(_d),lambda1(_lambda1),lambda2(_lambda2),alpha(_alpha),beta(_beta)
        {
            if(_d<1)
            {
                throw "CFtrlAlgo,bad _d!!!";//////析构函数中抛异常,要小心
            }
            else
            {
                w = new double[d]();//////其实这里可以不用保存w，但是为了不要频繁malloc和free，所以这里浪费些内存。
                z = new double[d]();//////初始化为全0,意义合理
                n =  new double[d]();/////初始化为全0,意义合理
            }
            train_calcloss_gap = _calcloss_gap;
            train_example_num = 0;
            train_logloss = 0;
        }
        inline double sigmod(double s)
        {
            ////////下面修正一下logistic的输入，不要太大和太小,太小和太大的浮点数精度很差，迭代多了误差会累积,影响后续结果
            if (s>35)
            {
                s = 35;
            }
            if(s<-35)
            {
                s =  -35;
            }
            return 1/(1+exp(-1*s));
        }
        /////计算一个样本在本训练对象的logstic函数值,注意传入的是特征的稀疏表示
        double logistic(m_FeatureItems& x)
        {
            double s = 0;
            for(m_FeatureItems::iterator iter= x.begin(); iter != x.end();iter++)
            {
                int i = iter->first;
                double val = iter->second;
                if(i<d)
                {
                    s+=val*w[i];
                }
                else
                {
                    return 0;
                }
            }
            return sigmod(s);
        }

        //////收到一个样本然后进行迭代,输入的是特征的稀疏表示
        //////y为0或者为1
        bool TrainAFeature(m_FeatureItems& x,int y)
        {
            double s = 0;
            for(m_FeatureItems::iterator iter= x.begin(); iter != x.end();iter++)
            {
                long i = iter->first;
                double val = iter->second;
                if(abs(z[i])<=lambda1)
                {
                    w[i]=0;
                }
                else
                {
                    w[i] = -(z[i]-sgn(z[i])*lambda1)/((beta+sqrt(n[i]))/alpha+lambda2);
                }
                s+=val*w[i];
            }
            double p = sigmod(s);////////这里也许有点问题，这里预测时也许不该用目前的训练得到的权重，而应该仅仅用本次新更新的特征的权重，
            ///////////////////////但其实没有影响，因为如果特征取值非0，那么他的权重一定被重新计算了，为0那么预测的时候相应的权重虽然不是本次迭代所得，但是不会起作用
            double g = 0;
            double sig =0 ;
            for(m_FeatureItems::iterator iter= x.begin(); iter != x.end();iter++)
            {
                int i = iter->first;
                double val = iter->second;
                g = (p-y)*val;//////当前损失函数的在本样本点处本维度的梯度
                sig= 1.0/alpha*(sqrt(n[i]+g*g)-sqrt(n[i]));
                z[i] += g-sig*w[i];
                n[i] += g*g;
            }

            //////打印logloss，因为aucloss不好在线计算(但是这个指标很准将用于离线参数选取)，所以这里采用logloss
            train_example_num++;
            train_logloss += logloss(p,y);
            if(train_example_num%train_calcloss_gap == 0)
            {
                cout<<"curloss: "<<train_logloss/train_example_num<<"\t"<<"train_example_num="<<train_example_num<<endl;
            }
                        
            return true;
        }

        bool dumpw(string& filename)
        {
            std::ofstream  ofile;
            ofile.open(filename.c_str());
            for(int i=0;i<d;i++)
            {
                if(w[i]>10e-10 || w[i]<-10e-10 )
                {
                    ofile<<i<<":"<<w[i]<<endl;
                }
            }
            ofile.close();
            return true;

        }
        
        /////析构函数
        ~CFtrlAlgorithm()
        {
            delete[] w;
            delete[] z;
            delete[] n;
        }
    private:
        ////////////////////计算变量////////////////
        //////////////////按用double表示且特征个数为1000000，那么本对象占用空间不到22M//////////////////
        /////////////////本类实现确实有内存浪费(因为采用数组实现)，不过其实这个不算什么,百万级别的特征指定够用且内存消耗并不大，数组实现毕竟运算速度很快//////
        double* w;//////从开始到上一轮迭代结束后,模型的参数
        double* z;//////从开始到上一轮迭代结束后，累积梯度
        double* n;//////从开始到上一轮迭代结束后,累积梯度平方
        ////////////////////参数////////////////////
        long d;/////输入数据的特征个数
        double beta ;//////作者在文中建议为1
        double alpha;//////此参数需要根据具体的数据集仔细调整,不过发现有的实现将这里默认取值为0.15
        double lambda1;/////L1正则化权重
        double lambda2;/////L2正则化权重
        ////////自身监视
        long train_calcloss_gap;/////每隔多少个样本就输出一下到当前的测试损失
        long train_example_num;/////总共过了多少个训练样本
        double train_logloss;///////当前测试损失
    };
    
}

#endif
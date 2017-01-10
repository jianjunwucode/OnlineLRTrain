/*
* description: 
*        ����Per-Coordinate FTRL-Proximal �㷨��logistic�ع��ʵ��.�㷨��ϸ�ڲμ�����:
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
    long max_feature_num = 1048576;//�����������,���ǵķ�������������֧��500W������
    typedef map<long,double> m_FeatureItems;// ϡ��������ʾ,keyΪ��������,valueΪ����ֵ,���������Ǽ������������ٶ���[-1,1],���Գ�����û�����κ����ݱ�׼��

   /////���ź���
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
    //////����logloss
    inline double logloss(double p,double y)
    {
        if(p<10e-15)/////Ԥ�����ֵ��Ҫ����10e-15
        {
            p = 10e-15;
        }
        if(p>1-10e-15)/////Ԥ�������ҪС��1-10e-15
        {
            p=1-10e-15; 
        }
        return  -y*log(p)-(1-y)*log(1-p);
    }

    ////////ģ��ѵ����,
    ///////ע�Ȿѵ����ֻ���ڸ��������������������ʹ��,���Ҳ����������и��ʲ���(����Ӧ�����������).���Ҫ�Ա������������߲�ͬ�Ĳ������ú�ģ�͵�Ч����
    ///////��ôĿǰ�����������ķ�ʽ(��ʵֻ��������ģ�Ͳ���ͬһ̨������ʱ�����Ҫ����������ͬһ̨�����Ļ���Ҫserver�Լ�����Э���󿽱��ַ�)
    ///////������Ҫ���ν�������ѡȡ�����ν������߶��������󣬸��ݵ����Ĳ�ͬ���ݺͲ���ָ���Ӹ���ͬ�Ķ���
    class CFtrlAlgorithm
    {
    public:
        CFtrlAlgorithm(long _d=max_feature_num,double _lambda1=1,double _lambda2=1,double _alpha=0.1,int _calcloss_gap=100,double _beta=1)
            :d(_d),lambda1(_lambda1),lambda2(_lambda2),alpha(_alpha),beta(_beta)
        {
            if(_d<1)
            {
                throw "CFtrlAlgo,bad _d!!!";//////�������������쳣,ҪС��
            }
            else
            {
                w = new double[d]();//////��ʵ������Բ��ñ���w������Ϊ�˲�ҪƵ��malloc��free�����������˷�Щ�ڴ档
                z = new double[d]();//////��ʼ��Ϊȫ0,�������
                n =  new double[d]();/////��ʼ��Ϊȫ0,�������
            }
            train_calcloss_gap = _calcloss_gap;
            train_example_num = 0;
            train_logloss = 0;
        }
        inline double sigmod(double s)
        {
            ////////��������һ��logistic�����룬��Ҫ̫���̫С,̫С��̫��ĸ��������Ⱥܲ�������������ۻ�,Ӱ��������
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
        /////����һ�������ڱ�ѵ�������logstic����ֵ,ע�⴫�����������ϡ���ʾ
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

        //////�յ�һ������Ȼ����е���,�������������ϡ���ʾ
        //////yΪ0����Ϊ1
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
            double p = sigmod(s);////////����Ҳ���е����⣬����Ԥ��ʱҲ������Ŀǰ��ѵ���õ���Ȩ�أ���Ӧ�ý����ñ����¸��µ�������Ȩ�أ�
            ///////////////////////����ʵû��Ӱ�죬��Ϊ�������ȡֵ��0����ô����Ȩ��һ�������¼����ˣ�Ϊ0��ôԤ���ʱ����Ӧ��Ȩ����Ȼ���Ǳ��ε������ã����ǲ���������
            double g = 0;
            double sig =0 ;
            for(m_FeatureItems::iterator iter= x.begin(); iter != x.end();iter++)
            {
                int i = iter->first;
                double val = iter->second;
                g = (p-y)*val;//////��ǰ��ʧ�������ڱ������㴦��ά�ȵ��ݶ�
                sig= 1.0/alpha*(sqrt(n[i]+g*g)-sqrt(n[i]));
                z[i] += g-sig*w[i];
                n[i] += g*g;
            }

            //////��ӡlogloss����Ϊaucloss�������߼���(�������ָ���׼���������߲���ѡȡ)�������������logloss
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
        
        /////��������
        ~CFtrlAlgorithm()
        {
            delete[] w;
            delete[] z;
            delete[] n;
        }
    private:
        ////////////////////�������////////////////
        //////////////////����double��ʾ����������Ϊ1000000����ô������ռ�ÿռ䲻��22M//////////////////
        /////////////////����ʵ��ȷʵ���ڴ��˷�(��Ϊ��������ʵ��)��������ʵ�������ʲô,���򼶱������ָ���������ڴ����Ĳ���������ʵ�ֱϾ������ٶȺܿ�//////
        double* w;//////�ӿ�ʼ����һ�ֵ���������,ģ�͵Ĳ���
        double* z;//////�ӿ�ʼ����һ�ֵ����������ۻ��ݶ�
        double* n;//////�ӿ�ʼ����һ�ֵ���������,�ۻ��ݶ�ƽ��
        ////////////////////����////////////////////
        long d;/////�������ݵ���������
        double beta ;//////���������н���Ϊ1
        double alpha;//////�˲�����Ҫ���ݾ�������ݼ���ϸ����,���������е�ʵ�ֽ�����Ĭ��ȡֵΪ0.15
        double lambda1;/////L1����Ȩ��
        double lambda2;/////L2����Ȩ��
        ////////�������
        long train_calcloss_gap;/////ÿ�����ٸ����������һ�µ���ǰ�Ĳ�����ʧ
        long train_example_num;/////�ܹ����˶��ٸ�ѵ������
        double train_logloss;///////��ǰ������ʧ
    };
    
}

#endif
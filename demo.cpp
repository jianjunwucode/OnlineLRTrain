#include "ftrl_proximal.hpp"
#include <vector>
#include <sstream>

using namespace std;
using namespace FTRL;

void SplitString(const std::string& strSrc, const  std::string strSpliter, std::vector<std::string>& vecStrs,bool bReserveNullString=false)
{
    vecStrs.clear();

    if (strSrc.empty() || strSpliter.empty()) return;

    size_t nStartPos=0;
    size_t nPos = strSrc.find(strSpliter);
    std::string strTemp;

    while (std::string::npos != nPos)
    {
    strTemp = strSrc.substr(nStartPos,nPos-nStartPos);
    if ((!strTemp.empty()) || (bReserveNullString && strTemp.empty()))
    vecStrs.push_back(strTemp);

    nStartPos = nPos+strSpliter.length();
    nPos = strSrc.find(strSpliter, nStartPos);
    }

    if(nStartPos != strSrc.size())
    {
    strTemp = strSrc.substr(nStartPos);
    vecStrs.push_back(strTemp);
    }
}

int main()
{
    CFtrlAlgorithm oFtrlAlgorithm;

    ///////开始流式训练
    ifstream  ifile;
    ifile.open("c_train.data");
    while(ifile.good())
    {

        char line[4096];
        ifile.getline(line,4096);
        string str_data(line);
        if(str_data.length()<10)
        {
            continue;
        }

        
        vector<string> v_str;
        SplitString(line,"\t",v_str);

        m_FeatureItems x;
        int y;
        for(int i=0;i<v_str.size();i++)
        {
            if(v_str[i].find("y")== string::npos)
            {
                int pos = v_str[i].find(":");
                long idx = stoi(v_str[i].substr(0,pos),NULL);
                double val = stoi(v_str[i].substr(pos+1,1),NULL);
                x[idx] = val;
            }
            else
            {
                int pos = v_str[i].find(":");
                y = stoi(v_str[i].substr(pos+1),NULL);
            }
        }
        oFtrlAlgorithm.TrainAFeature(x,y);

    }
    oFtrlAlgorithm.dumpw("result");
    ifile.close();



}
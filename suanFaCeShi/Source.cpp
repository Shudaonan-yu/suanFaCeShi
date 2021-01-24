#include <iostream>
#include"leetcode.h"
#include<stdlib.h>
#include"ContiguousList.h"
#include "jianZhiOffer.h"
#include<vector>
#include"linkedList.h"
using namespace std;

int Index2(string S, string T, int pos);//判断第pos个位置之后的子串长度
int getLength(int x);//判断整数位数
int main()
{/*
    string s, t;*/ 

#pragma region 力扣算法题库第389题
    /*leetcode test389;*/
    // cout << "请输入第一个小写字符串:";
  // cin >> s;
  // cout << "请输入第二个小写字符串:";
  // cin >> t;
  ///* cout << "不同的数字为：" << test389.findthedifference(s, t) << endl;*/
   /*cout << "不同的数字为：" << test389.findthedifference2(s, t) << endl;*/
#pragma endregion

#pragma region 括号匹配问题
   //cout << "请输入括号组合:";
    // cin >> s;
    // leetcode testkuohao;
    // //cout << testkuohao.isValid(s) << endl;
    // cout << testkuohao.isVaild2(s) << endl;
#pragma endregion

    
#pragma region 面试题08.06 汉诺塔问题
    jianZhiOffer hrt;
  /*  int n = 0;
    printf("请输入一个正整数：\n");
    scanf_s("%d", &n);
    if (n < 0) {
        printf("错误，请输入一个正整数：\n");
        return 0;
    }
        hrt.i= 1;
        hrt.hanNuo(n, 'A', 'B', 'C');
        printf("最后的总步数为%d\n", hrt.i - 1);
        return hrt.i - 1;*/
    //}
   /* while (cin >> num[i]) {
        i++;
    }*/
  /*  vector<int>A;
    vector<int>B;
    vector<int>C;
	int num;
	while(cin >> num)
	{
		if (cin.get() == '\n') {
			A.push_back(num);
			break;
		}
		A.push_back(num);
	}
    if(A.back() != NULL) {
		int i = A.size();
        hrt.hanNuo_Stack(i,A,B,C);
        vector<int>::iterator it;
		for (it = C.begin(); it != C.end(); it++)
			cout << *it << " ";
    }*/
    
    
#pragma endregion

#pragma region 反转链表
	int lo[4] = { 3,2,5,7 };
	linkedList::aLinkList L;
	linkedList tem;
	tem.createListF(L, lo, 4);
	linkedList::aNode* pre = tem.reverseLinklist(L);
	
#pragma endregion



#pragma region 顺序表操作
     /*   ContiguousList::SqList L;
        ContiguousList::InitList_dymantic(L);
        ContiguousList::IncreaseSize(L, 5);*/
       
        
#pragma endregion



#pragma region 无重复字符串
	string s;
	//cin >> s;
	leetcode test;
	//cout<<test.lengthOfLongestSubstring_byMap(s)<<endl;
	int kk = Index2("abcdedbcf", "bcf", 1);
#pragma endregion
	int lll = getLength(100);
	int a = 1.5 / 2;

	/*vector<int> t = { 2,5,6,0,0,1,2 };
	vector<int> t2 = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1 };
	bool flag = test.search(t2, 2);*/
	/*vector < vector<int> > t = { {1,4,7,11,15},{2,5,8,12,19},{3,6,9,16,22},{10,13,14,17,24},{18,21,23,26,30}};
	bool flag = test.searchMatrixEnhance(t, 5);*/
	vector<int> t = { 1,2,3,1,1};
	/*int fla = test.findmin(t);*/
	/*int te = test.minEatingSpeed(t, 8);*/
	//int te = test.shipWithinDays(t, 5);
	/*int t = test.multiply(-2, 3);*/
	vector<int> jk = { -2,0,0,2,2 };
	vector<vector<int> > ans = test.threeSum(jk);




	system("pause");
    return 0;
}

int Index2(string S, string T, int pos)
{
	int i, j;
	if (1 <= pos && pos <= S.length())
	{
		i = pos;//从主串的第pos个字符开始和子串的第一个字符比较
		j = 0;
		while (i <= S.length() && j < T.length())
		{
			if (S[i] == T[j]) // 若找到主串字符与子串首字符相等时继续比较后继字符
			{
				++i;
				++j;
			}
			else // 指针后退，主串字符下标后移一位，重新开始匹配
			{
				i = i + 1;
				j = 0;
			}
		}
		if (j == T.length())//条件成立，子串在主串中
			return i - T.length() + 1;//因为i是下标，下标编程序号需要+1
		else
			return 0;
	}
	else
		return 0;
}

int getLength(int x) {
	int len = 0;
	while (x) {
		x /= 10;
		++len;
	}
	return len;
}
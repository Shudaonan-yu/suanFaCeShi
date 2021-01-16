#include<string>
#include "ContiguousList.h"
using namespace std;
//void ContiguousList::InitList(SqList& L)
//{
//	for (int i = 0; i < MaxSize; i++) L.data[i] = 0;
//	L.length = 0;
//}

void ContiguousList::InitList_dymantic(SqList& L) //动态化初始线性表存储空间
{
	L.data = (int*)malloc(InitSize * sizeof(int));
	L.length = 0;
	L.MaxSzie = InitSize;
}

void ContiguousList::IncreaseSize(SqList& L, int len)
{
	int* p = L.data;
	L.data = (int*)malloc(sizeof(int) * (len + L.MaxSzie)); //另辟战场，不是在原来的内存空间后新加一段，而是重新申请了一片内存空间
	for (int i = 0; i < L.length; i++) {
		L.data[i] = p[i]; //把数据复制到新区域
	}
	L.MaxSzie = L.MaxSzie + len;
	free(p);
}


ContiguousList::ContiguousList()
{
}

ContiguousList::~ContiguousList()
{
}

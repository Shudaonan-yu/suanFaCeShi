#include<string>
#include "ContiguousList.h"
using namespace std;
//void ContiguousList::InitList(SqList& L)
//{
//	for (int i = 0; i < MaxSize; i++) L.data[i] = 0;
//	L.length = 0;
//}

void ContiguousList::InitList_dymantic(SqList& L) //��̬����ʼ���Ա�洢�ռ�
{
	L.data = (int*)malloc(InitSize * sizeof(int));
	L.length = 0;
	L.MaxSzie = InitSize;
}

void ContiguousList::IncreaseSize(SqList& L, int len)
{
	int* p = L.data;
	L.data = (int*)malloc(sizeof(int) * (len + L.MaxSzie)); //���ս����������ԭ�����ڴ�ռ���¼�һ�Σ���������������һƬ�ڴ�ռ�
	for (int i = 0; i < L.length; i++) {
		L.data[i] = p[i]; //�����ݸ��Ƶ�������
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

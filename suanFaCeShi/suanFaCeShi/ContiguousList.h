#pragma once
class ContiguousList
{
public:

#pragma region ˳������
//��̬����
//#define MaxSize 10
//	typedef struct {
//		int data[MaxSize];
//		int length;
//	}SqList;
//void static InitList(SqList& L);
	//��̬����
#define InitSize 10
	typedef struct {
		int* data; //ָʾ��̬���������ָ��
		int MaxSzie;//˳�����������
		int length;//˳���ĵ��ڳ���
	}SqList;
	// ��ʼ��һ��˳���,�ö�̬����
	void static InitList_dymantic(SqList& L);

	//���Ӷ�̬����ĳ���
	void static IncreaseSize(SqList& L, int len);
		


#pragma endregion

	
	ContiguousList();
	~ContiguousList();
private:

};


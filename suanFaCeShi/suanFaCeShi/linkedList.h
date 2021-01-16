#pragma once
#include <string>
class linkedList
{
public:

	//������ṹ
	typedef struct aNode {
		int data;
		struct aNode *next;
		aNode(int val) :data(val), next(NULL) {}
	}aNode, *aLinkList;
	void initAlinkList(aLinkList &L);//����һ��ͷ���

	void createListF(aLinkList &L, int a[], int n);//ͷ�巨����������
	
	void createListR(aLinkList &L, int a[], int n);//β�巨����������;

	aNode* reverseLinklist(aLinkList &L);//��ת����

	aNode* removeElements(aNode* head,int val);//�Ƴ�������ָ��valֵ�Ľ��

	aNode* removeElements_byDuumyHead(aNode* head, int val);//ʹ������ͷ���ķ�ʽ�Ƴ�



	//˫����ṹ
	typedef struct dNode {
		double date;
		struct dNode* prior, * next;
	}dNode, *dLinkList;

	bool initDlinList(dLinkList& L);
private:
	
};


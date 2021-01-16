#pragma once
#include <string>
class linkedList
{
public:

	//单链表结构
	typedef struct aNode {
		int data;
		struct aNode *next;
		aNode(int val) :data(val), next(NULL) {}
	}aNode, *aLinkList;
	void initAlinkList(aLinkList &L);//创建一个头结点

	void createListF(aLinkList &L, int a[], int n);//头插法建立单链表
	
	void createListR(aLinkList &L, int a[], int n);//尾插法建立单链表;

	aNode* reverseLinklist(aLinkList &L);//反转链表

	aNode* removeElements(aNode* head,int val);//移除链表中指定val值的结点

	aNode* removeElements_byDuumyHead(aNode* head, int val);//使用虚拟头结点的方式移除



	//双链表结构
	typedef struct dNode {
		double date;
		struct dNode* prior, * next;
	}dNode, *dLinkList;

	bool initDlinList(dLinkList& L);
private:
	
};


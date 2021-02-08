#include "linkedList.h"
#include<stdlib.h>
void linkedList::initAlinkList(aLinkList & L)//��ʼ����������
{
	L = (aNode*)malloc(sizeof(aNode));
	L->next = NULL;
}

void linkedList::createListF(aLinkList &L, int a[], int n)//������������ ������ķ�ʽ��ͷ�巨
{
	aNode* s;
	initAlinkList(L);
	for (int i = 0; i < n; i++) {
		s = (aNode*)malloc(sizeof(aNode));
		s->data = a[i];
		s->next = L->next;
		L->next = s;
	}
}


void linkedList::createListR(aLinkList &L, int a[], int n)//β�巨
{
	initAlinkList(L);
	aNode* R = L;
	for (int i = 0; i < n; i++) {
		aNode* s = (aNode*)malloc(sizeof(aNode));
		s->data = a[i];
		R->next = s;
		R = s;			
	}
	R->next == NULL;
}
//��ת����
linkedList::aNode* linkedList::reverseLinklist(aLinkList &L)
{   
	aNode* pre = nullptr;
	aNode* cur = L;
	aNode* next = nullptr;
 
	while (cur) {
		next = cur->next;
		cur->next = pre;
		pre = cur;
		cur = next;
	}
	return pre;
}

linkedList::aNode * linkedList::removeElements(aNode* head, int val)
{
	//ɾ��ͷ���
	while (head != NULL && head->data == val) {
		aNode* tmp = head; //�ݴ���head.�������Ϊhead���ָ�뱣���˺������нڵ㣬������Ӳ��ܶ�
		head = head->next;
		delete tmp;
	}
	//ɾ����ͷ���
	aNode* cur = head;
	while (cur != NULL && cur->next != NULL) {
		if (cur->next->data == val) {
			aNode* tmp = cur->next; //�ڹҿ�����ʱһ��Ҫ�ݴ�Ҫɾ���Ľ�㣬��Ȼ���Ҳ�������
			cur->next = cur->next->next;
			delete tmp;
		}
		else
		{
			cur = cur->next;
		}
	}

	return head;
}

linkedList::aNode * linkedList::removeElements_byDuumyHead(aNode * head, int val)
{
	aNode* dummyHead = new aNode(0);//����һ������ͷ���
	dummyHead->next = head;//������ͷ���ĺ�̽��ָΪhead
	aNode* cur = dummyHead;//��Ҫ����һ����ָ��
	while (cur->next != NULL)
	{
		if (cur->next->data == val) {
			aNode* tmp = cur->next;
			cur->next = cur->next->next;
			delete tmp;
		}
		else
		{
			cur = cur->next;
		}
	}

	return dummyHead->next;//�ǵ��Ƿ�������ͷ������һ��
}

bool linkedList::initDlinList(dLinkList& L)
{
    L = (dNode*)malloc(sizeof(dNode));//����һ��ͷ���

    return false;
}

#include "linkedList.h"
#include<stdlib.h>
void linkedList::initAlinkList(aLinkList & L)//初始化单向链表
{
	L = (aNode*)malloc(sizeof(aNode));
	L->next = NULL;
}

void linkedList::createListF(aLinkList &L, int a[], int n)//建立单向链表 用数组的方式，头插法
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


void linkedList::createListR(aLinkList &L, int a[], int n)//尾插法
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
//反转链表
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
	//删除头结点
	while (head != NULL && head->data == val) {
		aNode* tmp = head; //暂存下head.可以理解为head这个指针保存了后续所有节点，这个链接不能断
		head = head->next;
		delete tmp;
	}
	//删除非头结点
	aNode* cur = head;
	while (cur != NULL && cur->next != NULL) {
		if (cur->next->data == val) {
			aNode* tmp = cur->next; //在挂开链接时一定要暂存要删除的结点，不然就找不到它了
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
	aNode* dummyHead = new aNode(0);//设置一个虚拟头结点
	dummyHead->next = head;//将虚拟头结点的后继结点指为head
	aNode* cur = dummyHead;//总要创建一个现指针
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

	return dummyHead->next;//记得是返回虚拟头结点的下一个
}

bool linkedList::initDlinList(dLinkList& L)
{
    L = (dNode*)malloc(sizeof(dNode));//分配一个头结点

    return false;
}

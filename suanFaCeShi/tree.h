#pragma once
#include<iostream>
#include<string>
#include<assert.h>
#ifndef tree_H
#define tree_H


class tree
{
	//�������������
	
	typedef struct threadNode {
		int data;
		struct threadNode* lchild, * rchild;
		int ltag, rtag;//����������־��Ϊ0ʱ��־������ӽ��洢������������ṹ�ĺ��ӽ�㣬1��ʱ���ʾ�������ض����������е��߼�ǰ���ͺ���
	}threadNode,*threadTree;

#pragma region ����������������


	//����������������T
	bool createInThread(threadTree T);


	//���������������һ�߱���һ��������
	void inThread(threadTree T,threadNode* &pre);

	//���ʽ��ʱִ�еĲ������˴�Ϊ��������,����ָ�룬���⸴�ƴ�������
	void visit(threadNode* q,threadNode* &pre);
	

	//�ҵ���PΪ�������������У���һ�����߼���������Ľ�㣬���մ�ʩ
	threadNode* firstNode(threadNode* p);

	//�������������������ҵ����p�ĺ�̽��
	threadNode* findNextNode(threadNode* p);

	//�����������������������������������������ʵ�ֵķǵݹ��㷨)
	void inOrder(threadNode* T);


	//�ҵ���PΪ���������У����һ������������Ľ��
	threadNode* lastNode(threadNode* p);


	//�����������������У���p����ǰ�����
	threadNode* findPreNode(threadNode* p);

	//�������������������������������
	void revInOrder(threadNode* T);
#pragma endregion



//�������������������ҵ����P�������̽��,���rtag��Ϊ0�����������������������������̽��������ӣ���û�������������̽���Ϊ�Һ���
	threadNode* findNextNodeInPreOrder(threadNode* p);




};

#endif // !tree_H
#pragma once
#include<iostream>
#include<string>
#include<assert.h>
#ifndef tree_H
#define tree_H


class tree
{
	//线索二叉树结点
	
	typedef struct threadNode {
		int data;
		struct threadNode* lchild, * rchild;
		int ltag, rtag;//左，右线索标志，为0时标志这个孩子结点存储的是树的物理结构的孩子结点，1的时候表示的是在特定遍历序列中的逻辑前驱和后驱
	}threadNode,*threadTree;

#pragma region 中序线索化二叉树


	//中序线索化二叉树T
	bool createInThread(threadTree T);


	//中序遍历二叉树，一边遍历一边线索化
	void inThread(threadTree T,threadNode* &pre);

	//访问结点时执行的操作，此处为建立线索,传入指针，避免复制大数据域
	void visit(threadNode* q,threadNode* &pre);
	

	//找到以P为根的物理子树中，第一个被逻辑中序遍历的结点，保险措施
	threadNode* firstNode(threadNode* p);

	//在中序线索二叉树中找到结点p的后继结点
	threadNode* findNextNode(threadNode* p);

	//对中序线索二叉树进行正序中序遍历（利用线索实现的非递归算法)
	void inOrder(threadNode* T);


	//找到以P为根的子树中，最后一个被中序遍历的结点
	threadNode* lastNode(threadNode* p);


	//在中序线索二叉树中，找p结点的前驱结点
	threadNode* findPreNode(threadNode* p);

	//对中序线索二叉树进行逆序中序遍历
	void revInOrder(threadNode* T);
#pragma endregion



//在先序线索二叉树中找到结点P的先序后继结点,如果rtag不为0，则存在右子树，若有左子树，后继结点就是左孩子，若没有左子树，则后继结点就为右孩子
	threadNode* findNextNodeInPreOrder(threadNode* p);




};

#endif // !tree_H
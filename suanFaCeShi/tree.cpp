#include "tree.h"
#include<string>
bool tree::createInThread(threadTree T)
{
    if (T == nullptr)
        return false;
    threadNode* pre = nullptr; //preָ���ʼʱָ�������ǰһ��������Ϊnull
    inThread(T, pre);
    //preָ������һ������غ�ʱ����������
    if (pre->rchild == NULL) {
        pre->rtag = 1;//������������һ�����
    }

}

void tree::inThread(threadTree T, threadNode* &pre)
{
    if (T != NULL) {//������ֹ����
        tree::inThread(T->lchild, pre);//�������������
        visit(T, pre);//���ʸ����
        tree::inThread(T->rchild, pre);
    }
}

void tree::visit(threadNode* q, threadNode*& pre)
{
    if (q->lchild != NULL) {
        q->lchild = pre;//��q�Ŀ�������ָ����Ϊǰ������
        q->ltag = 1;
    }
    //ǰ�����
    if (pre != NULL && pre->rchild == NULL) {
        pre->rchild = q;
        pre->rtag = 1;
    }
    pre = q;//ִ����ý��Ĳ�������preָ��ǰ���
}

tree::threadNode* tree::firstNode(threadNode* p)
{
    assert(p != NULL);
    //��������ṹ�����½��
    while (p->ltag == 0)
        p = p->lchild;
    return p;
}

tree::threadNode* tree::findNextNode(threadNode* p)
{
    assert(p != NULL);
    //���������������½��
    if (p->rtag == 0)
        return  firstNode(p->rchild);
    else
        return p->rchild;//rtag == 1ֱ�ӷ��غ������
}

void tree::inOrder(threadNode* T)
{
    for (threadNode* cur = firstNode(T); cur != NULL; cur = findNextNode(cur)) {
        //visit(cur);
    }
     
}

tree::threadNode* tree::lastNode(threadNode* p)
{
    assert(p != NULL);
    //���������������½��
    while (p->rtag == 0) p = p->rchild;
    return p;
}

tree::threadNode* tree::findPreNode(threadNode* p)
{
    assert(p != NULL);
    //���������������½��
    if (p->ltag == 0)
        return lastNode(p->lchild);
    return p->lchild;
}

void tree::revInOrder(threadNode* T)
{
    for (threadNode* cur = lastNode(T); cur != NULL; cur = findPreNode(cur)) {
       // visit(cur);
    }
}

tree::threadNode* tree::findNextNodeInPreOrder(threadNode* p)
{
    assert(p != NULL);
   /* if (p->rtag == 1) return p->rchild;
    else if (p->ltag == 1) {
        return  p->rchild;
    }
    else return p->lchild;*/

    if (p->ltag == 0) {
        return  p->rchild;
    }
    else return p->lchild;
}
        


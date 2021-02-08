#include "tree.h"
#include<string>
bool tree::createInThread(threadTree T)
{
    if (T == nullptr)
        return false;
    threadNode* pre = nullptr; //pre指针初始时指向根结点的前一个，所以为null
    inThread(T, pre);
    //pre指针和最后一个结点重合时，遍历结束
    if (pre->rchild == NULL) {
        pre->rtag = 1;//处理遍历的最后一个结点
    }

}

void tree::inThread(threadTree T, threadNode* &pre)
{
    if (T != NULL) {//遍历终止条件
        tree::inThread(T->lchild, pre);//中序遍历左子树
        visit(T, pre);//访问根结点
        tree::inThread(T->rchild, pre);
    }
}

void tree::visit(threadNode* q, threadNode*& pre)
{
    if (q->lchild != NULL) {
        q->lchild = pre;//让q的空闲左孩子指针作为前驱线索
        q->ltag = 1;
    }
    //前驱后继
    if (pre != NULL && pre->rchild == NULL) {
        pre->rchild = q;
        pre->rtag = 1;
    }
    pre = q;//执行完该结点的操作后，让pre指向当前结点
}

tree::threadNode* tree::firstNode(threadNode* p)
{
    assert(p != NULL);
    //树的物理结构最左下结点
    while (p->ltag == 0)
        p = p->lchild;
    return p;
}

tree::threadNode* tree::findNextNode(threadNode* p)
{
    assert(p != NULL);
    //找右子树的最左下结点
    if (p->rtag == 0)
        return  firstNode(p->rchild);
    else
        return p->rchild;//rtag == 1直接返回后继线索
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
    //找树的物理最右下结点
    while (p->rtag == 0) p = p->rchild;
    return p;
}

tree::threadNode* tree::findPreNode(threadNode* p)
{
    assert(p != NULL);
    //找左子树的最右下结点
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
        


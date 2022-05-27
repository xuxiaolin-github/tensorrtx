#pragma once

#include<iostream>
#include <mutex>
//#include <queue>
#include <condition_variable>

using namespace std;

#define DEFAULT_SIZE 10

template<typename T>
class CircuQueue
{

private:
    std::mutex mux;
    std::condition_variable cond_en;
    std::condition_variable cond_de;

    int m_front = 0;
    int m_rear = 0;
    int m_size = DEFAULT_SIZE;
    bool m_exit = false;
    T *m_data;

public:
    CircuQueue() {
	m_data = new T[m_size];
    }

    CircuQueue(int size) {
	m_size = size;
	m_data = new T[m_size];
    }

    bool enqueue(T &data) {
	std::unique_lock<std::mutex> lk(mux);
	cond_en.wait( lk,
		      [this] {
			  return !isFull();
		      }
	    );

	m_data[m_rear] = data;
	//cout<< &m_data[m_rear] << " " << &data <<endl;
	m_rear = (m_rear+1) % m_size;

	cond_de.notify_one();
	return true;
    }

    bool dequeue(T& data) {
	std::unique_lock<std::mutex> lk(mux);

	cond_de.wait( lk,
		      [this] {
			  if (m_exit) {
			      return true;
			  }
			  return !isEmpty();
		      }
	    );
//    cond_de.wait(lk);

	if( m_exit || isEmpty() ) {
	    return false;
	}

//    if(isEmpty())
//    {
//	cout<<"The queue is empty!"<<endl;
//	return false;
//    }

	data = m_data[m_front];
	m_front = (m_front+1) % m_size;

	cond_en.notify_one();
	return true;
    }

    bool isFull() {
	if((m_rear+1) % m_size == m_front) {
	    //cout<<"+++ The queue is full!"<<endl;
	    return true;
	}
	return false;
    }

    bool isEmpty() {
	if(m_rear == m_front) {
	    //cout<<"+++ The queue is empty!"<<endl;
	    return true;
	}
	return false;
    }

    ~CircuQueue() {
	delete [] m_data;
    }

    void exit() {
	std::lock_guard<std::mutex> lk(mux);
	m_exit = true;
	cond_de.notify_all();
    };
};


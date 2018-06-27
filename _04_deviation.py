import threading
try: 
    import Queue
except ImportError: 
    import queue as Queue

#SVR - PREDICT AVERAGE DIFFERENCE OF MULTIPLE KERNELS 
'''
DEPRICATED....
def deviation_avg_SVR(dates, prices, lin_test_predictions, poly_test_predictions, rbf_test_predictions):

    que = Queue.Queue(4) #Needed to save results from the threads

    t1 = threading.Thread(target=lambda q, arg1: q.put(deviation_avg_single(dates, prices, lin_test_predictions, )), args=(que, 'Threads')) #Calculating the average in multiple threads for better performance
    t2 = threading.Thread(target=lambda q, arg1: q.put(deviation_avg_single(dates, prices, poly_test_predictions, )), args=(que, 'Threads'))
    t3 = threading.Thread(target=lambda q, arg1: q.put(deviation_avg_single(dates, prices, rbf_test_predictions, )), args=(que, 'Threads'))

    #Starts a thread for each modell, so that they get computed simuntainously
    t1.start()
    print('t1 started')
    t2.start()
    print('t2 started')
    t3.start()
    print('t3 started')
 
    #Synchronize the threads
    t1.join()
    print('Lin Average: ', que.get())
    t2.join()
    print('Poly Average: ', que.get())
    t3.join()
    print('Rbf Average: ', que.get())
    return
'''

#PREDICT AVERAGE DIFFERENCE
def deviation_avg_single (dates, prices, test_predictions):
    counter = 0
    difference = 0
    for prediction in test_predictions: 
        actual = get_price_at_date(dates, prices, prediction[0])
        my_prediction = prediction[1]
        difference = difference + abs(my_prediction - actual) 
        counter += 1 #iterate counter
        
    return difference/counter #avg difference

#PREDICT SMAPEx
def smape_and_smdape (dates, prices, test_predictions):
    counter = 0
    difference = 0
    smape = 0
    sapes = []

    for prediction in test_predictions: 
        actual = get_price_at_date(dates, prices, prediction[0])
        my_prediction = prediction[1]
        difference = difference + abs(my_prediction - actual) 
        counter += 1 #iterate counter
        smape += abs(my_prediction - actual)/((abs(actual)+abs(my_prediction))/2)
        sapes.append(abs(my_prediction - actual)/((abs(actual)+abs(my_prediction))/2))

    smdape = [int(len(sapes)/2)]

    return 100/float(counter) * smape, smdape #avg difference

#function that returns the price for a given date
def get_price_at_date(dates, prices, date):
    my_index = dates.tolist().index(date)
    return prices[my_index]

      
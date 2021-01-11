import os
import re
import time
import numpy as np
import numpy.ma as npma
import math as m
import pandas as pd
import seaborn as sns
import datetime as dt
import random as rdn
import matplotlib.pyplot as plt
import scipy
plt.style.use('ggplot')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

class DrawPlotByTS():
    def __init__(self):
        print('Initializing DrawPlotByTS...')
        print('Sucessfully Initializing DrawPlotByTS')
    
    def PlotLogPicture(self,dfname,period,*input_return): #list,single_df
        fig = plt.figure(figsize=(15,3))
        plt.title("price")
        plt.yscale("log")
        col_labels=['return','std','sharp','max_drawdown']
        row_labels=dfname
        table_vals=[]
        input_len=len(input_return)
        time=input_return[0].index
        for i,single in enumerate(input_return):
            cumulate=single.cumprod()
            drawdown=self.CalculateDrawdown(cumulate)
            [a,b,c,d]=self.CalculateBasicInformation(cumulate,single,period)[0:4]
            table_vals.append([a,b,c,d])
            plt.plot(time,cumulate,label=dfname[i])
        plt.legend()
        plt.table(cellText=table_vals,rowLabels=row_labels,colLabels=col_labels,loc='lower right',colWidths = [0.15]*4)
        plt.show()
        
    def PlotDrawdown(self,dfname,*cumulate): #list,cumulate_df
        input_len=len(cumulate)
        time=cumulate[0].index
        drawdown_combine=[]
        for i,cumulate_each in enumerate(cumulate):
            plt.figure(figsize=(15,3))
            plt.title(dfname[i])
            drawdown_combine.append(self.CalculateDrawdown(cumulate_each))
            plt.fill_between(time,0,drawdown_combine[-1],facecolor='red')
            plt.ylim(top = 0, bottom = -0.6)
            plt.show()
        if len(cumulate)==2:
            plt.figure(figsize=(15,3))
            plt.title("combine_drawdown")
            drawdown_temp=[]
            for i in range(len(drawdown_combine[0])):
                if drawdown_combine[0][i]>=drawdown_combine[1][i]:
                    drawdown_temp.append(drawdown_combine[0][i])
                else:
                    drawdown_temp.append(drawdown_combine[1][i])
            plt.fill_between(time,0,drawdown_combine[0],facecolor='blue',label=dfname[0])
            plt.fill_between(time,drawdown_temp,drawdown_combine[1],facecolor='red',label=dfname[1])
            plt.ylim(top = 0, bottom = -0.6)
            plt.legend(loc='lower right')
            plt.show()
    def CalculateDrawdown(self,price):
        drawdown=0
        x=1
        for i in range(1,len(price)):
            if price.iloc[i]>x:
                x=price.iloc[i]
            drawdown=np.append(drawdown,-(1-price.iloc[i]/x))
        return drawdown
    
    def TwoDigitsPercent(self,input_float):
        return ('%s'%  (np.round(100*input_float,2))+'%')
    
    def TwoDigits(self,input_float):
        return ('%s'%  (np.round(input_float,2)))
    
    def CalculateBasicInformation(self,price,ret,data_period):
        data_period=float(data_period)
        print(price.iloc[-1],data_period,price.shape[0])
        exp_return=(price.iloc[-1]**(data_period/price.shape[0])-1.0) #imply that price[0]=1
        std=ret.std()*m.sqrt(data_period)
        sharpe=exp_return/std
        max_drawdown=pd.DataFrame(self.CalculateDrawdown(price)).min().iloc[0]
        return_dd_ratio = -exp_return/max_drawdown
        outputs = [exp_return,std,sharpe,max_drawdown,return_dd_ratio]
        outputs_str = []
        for o in outputs:
            if o<1:
                outputs_str.append(self.TwoDigitsPercent(o))
            else:
                outputs_str.append(self.TwoDigits(o))
        return outputs_str

    def CalculateBasicInformationFast(self,price,ret,data_period):
        data_period=float(data_period)
        #print(price.iloc[-1],data_period,price.shape[0])
        exp_return=(price.iloc[-1]**(data_period/price.shape[0])-1.0) #imply that price[0]=1
        std=ret.std()*m.sqrt(data_period)
        sharpe=exp_return/std
        max_drawdown=pd.DataFrame(self.CalculateDrawdown(price)).min().iloc[0]
        return_dd_ratio = -exp_return/max_drawdown
        outputs = [exp_return,std,sharpe,max_drawdown,return_dd_ratio]
        return outputs

class Activate():
    def BinaryStep(self,df,interval,unact_value=0,include_bound=True):
        threshold_low,threshold_up = interval[0],interval[1]
        if include_bound == True:
            res = np.where((df>=threshold_low)&(df<=threshold_up),1,unact_value)
        else:
            res = np.where((df>threshold_low)&(df<threshold_up),1,unact_value)
        res = self.TransNPtoOriDF(res,df)
        return res

    def Sigmoid(self,df):
        res = 1/(1 + np.exp(-df))
        return res

    def Tanh(self,df):
        res = np.tanh(df)
        return res

    def RELU(self,df,threshold):
        res = np.where(df>threshold,df,0)
        res = self.TransNPtoOriDF(res,df)
        return res

    def LeakyRELU(self,df,threshold,slope):
        res = np.where(df>threshold,df,coef*df)
        res = self.TransNPtoOriDF(res,df)
        return res

    def Softmax(self,df):
        df_exp = np.exp(df)
        exp_sum = np.sum(df_exp,axis=1)
        df_exp_sum = np.tile(exp_sum,reps=(len(df.columns),1)).T
        res = df_exp/df_exp_sum
        return res

    def Swish(self,df,threshold):
        res = df/(1 + np.exp(-df))
        return res

class Operators(Activate):
    def __init__(self):
        print('Initializing Operator...')
        print('Successfuly Initializing Operator')

    def GetDFShell(self,df,value=0):
        res = np.full_like(df,value)
        res = pd.DataFrame(res,index = df.index,columns=df.columns)
        return res
    
    def TransDFtoNP(self,matrix):
        res = matrix.copy()
        if type(matrix) == pd.DataFrame:
            res = matrix.values
        return res
        
    def TransNPtoOriDF(self,np_array,ori_matrix):
        res = np_array.copy()
        if type(ori_matrix) == pd.DataFrame:
            res = pd.DataFrame(res,index=ori_matrix.index,columns=ori_matrix.columns)
        return res
    
    def FillRowN(self,df,fill_in_content,row_number):
        res,fill_array = self.TransDFtoNP(df),self.TransDFtoNP(fill_in_content)
        res[row_number,:] = fill_array[row_number,:]
        res = self.TransNPtoOriDF(res,df)
        return res
    
    def FwdChangeRate(self,df,period=1):
        df_next = df.shift(-period)
        return df_next/df - 1
        
    def BwdChangeRate(self,df,period=1):
        df_last = df.shift(period)
        return df/df_last - 1
    
    def FwdChangeNumber(self,df,period=1):
        df_next = df.shift(-period)
        return df_next - df
        
    def BwdChangeNumber(self,df,period=1):
        df_last = df.shift(period)
        return df - df_last

    def GetSegList(self,segment_np):
        s_list = []
        for s in (segment_np[[0,-1],:].flatten()):
            if s==s and s not in s_list:
                s_list.append(s)
        return s_list

    def SegFirmsCount(self,segment_df):
        segment_np = segment_df.values
        seglist = self.GetSegList(segment_np)
        res = np.full_like(segment_df,0)
        print(len(seglist))
        for seg in seglist:
            seg_boolin = np.where(segment_np==seg,1,0)
            seg_boolin = self.TransNPtoOriDF(seg_boolin,segment_df)
            seg_num = self.Extend1DTSto2D(np.sum(seg_boolin,axis=1),seg_boolin)
            seg_num_masked = np.where(segment_np==seg,seg_num,0)
            res += seg_num_masked
        res = self.TransNPtoOriDF(res,segment_df)
        res = res.where(segment_df==segment_df)
        return res

    def SegSum(self,df,segment_df):
        segment_np,np_array = segment_df.values,df.values
        seglist = self.GetSegList(segment_np)
        res = np.full_like(df,0)
        for seg in seglist:
            seg_boolin = np.where(segment_np==seg,np_array,0)
            seg_boolin = self.TransNPtoOriDF(seg_boolin,df)
            seg_sum = self.Extend1DTSto2D(np.sum(seg_boolin,axis=1),seg_boolin)
            seg_sum_masked = np.where(segment_np==seg,seg_sum,0)
            res += seg_sum_masked
        res = self.TransNPtoOriDF(res,df)
        res = res.where(segment_df==segment_df)
        return res

    def SegMean(self,df,segment_df):
        segment_np,np_array = segment_df.values,df.values
        seglist = self.GetSegList(segment_np)
        res = np.full_like(df,0)
        for seg in seglist:
            seg_boolin = np.where(segment_np==seg,np_array,np.nan)
            seg_boolin = self.TransNPtoOriDF(seg_boolin,df)
            seg_mean = pd.DataFrame(np.nanmean(seg_boolin,axis=1),index=seg_boolin.index)
            seg_mean = self.Extend1DTSto2D(seg_mean,seg_boolin) 
            seg_mean_masked = np.where(segment_np==seg,seg_mean,0)
            res += seg_mean_masked
        res = self.TransNPtoOriDF(res,df)
        res = res.where(segment_df==segment_df)
        return res

    def SegRank(self,df,segment_df,max_has_maxrank=False):
        segment_np,np_array = segment_df.values,df.values
        seglist = self.GetSegList(segment_np)
        print(len(seglist))
        res = np.full_like(df,0)
        for seg in seglist:
            seg_boolin = np.where(segment_np==seg,np_array,np.nan)
            seg_boolin = self.TransNPtoOriDF(seg_boolin,df)
            seg_rank = self.Rank(seg_boolin,max_has_maxrank=max_has_maxrank)
            seg_rank_masked = np.where(segment_np==seg,seg_rank,0)
            res += seg_rank_masked
        res = self.TransNPtoOriDF(res,df)
        res = res.where(segment_df==segment_df)
        return res

    def SegSubByRankNData(self,df,segment_df,ranked_segment_df,n=1):
        np_array,segment_np,segment_rank_np = df.values,segment_df.values,ranked_segment_df.values
        seglist = self.GetSegList(segment_np)
        res = np.full_like(df,0)
        for i in range(df.shape[0]):
            for j1 in range(df.shape[1]):
                if segment_rank_np[i,j1] == 1:
                    for j2 in range(df.shape[1]):
                        if segment_np[i,j1] == segment_np[i,j2]:
                            res[i,j2] = np_array[i,j1]
        res = self.TransNPtoOriDF(res,df)
        res = res.where(segment_df==segment_df)
        return res


    def SegZscore(self,df,segment_df):
        segment_np,np_array = segment_df.values,df.values
        seglist = self.GetSegList(segment_np)
        res = np.full_like(df,0)
        for seg in seglist:
            seg_boolin = np.where(segment_np==seg,np_array,np.nan)
            seg_boolin = self.TransNPtoOriDF(seg_boolin,df)
            seg_rank = self.RowZscore(seg_boolin)
            seg_rank_masked = np.where(segment_np==seg,seg_rank,0)
            res += seg_rank_masked
        res = self.TransNPtoOriDF(res,df)
        res = res.where(segment_df==segment_df)
        return res

    def MovingAverage(self,df,period):
        np_array = self.TransDFtoNP(df)
        res = np_array.copy()
        for i in range(1,period):
            x = np.roll(np_array,i,axis=0)
            res += x
        res = res/period
        try:
            res[:period-1,:] = np.nan
        except:
            res[:period-1] = np.nan
        res = self.TransNPtoOriDF(res,df)
        return res

    def ExpMovingAverage(self,df,alpha): #EMAt = alpha*(today)+(1-alpha)*(EMAt-1)
        np_array = self.TransDFtoNP(df)
        res = np_array.copy()
        for i in range(1,len(np_array)):
            temp = res[i-1,:].copy()
            temp2 = np.where(temp==temp,temp,0)
            res[i,:] = (1-alpha)*temp2 + alpha*np_array[i,:]
        res = self.TransNPtoOriDF(res,df)
        return res

    def RowMean(self,df):
        np_array = self.TransDFtoNP(df)
        row_mean = np.nanmean(np_array,axis=1)
        row_mean = pd.DataFrame(row_mean,index=df.index)
        res = self.Extend1DTSto2D(row_mean,df)
        return res

    def RowStd(self,df):
        np_array = self.TransDFtoNP(df)
        row_std = np.nanstd(np_array,axis=1)
        row_std = pd.DataFrame(row_std,index=df.index)
        res = self.Extend1DTSto2D(row_std,df)
        return res

    def RowZscore(self,df):
        row_mean = self.RowMean(df)
        row_std = self.RowStd(df)
        res = (df - row_mean)/row_std
        return res

    def ColumnStack(self,df):
        np_array = self.TransDFtoNP(df)
        res = np.full_like(np_array,np.nan,dtype=np.float)
        for i in range(len(np_array)):
            temp = np.nansum(np_array[:i+1,:],axis=0)
            res[i,:] = temp
        res = self.TransNPtoOriDF(res,df)
        return res

    def ColumnRSV(self,df,period):
        np_array = self.TransDFtoNP(df)
        res = np.full_like(np_array,np.nan,dtype=np.float)
        for i in range(period-1,len(np_array)):
            temp = (np_array[i,:]-np.min(np_array[i-period+1:i+1,:],axis=0))/(np.max(np_array[i-period+1:i+1,:],axis=0)-np.min(np_array[i-period+1:i+1,:],axis=0))
            res[i,:] = temp
        res = self.TransNPtoOriDF(res,df)
        return res

    def ColumnSum(self,df,period):
        np_array = self.TransDFtoNP(df)
        res = np.full_like(np_array,np.nan,dtype=np.float)
        for i in range(period-1,len(np_array)):
            temp = np.sum(np_array[i-period+1:i+1,:],axis=0)
            res[i,:] = temp
        res = self.TransNPtoOriDF(res,df)
        return res

    def ColumnMean(self,df,period):
        np_array = self.TransDFtoNP(df)
        res = np.full_like(np_array,np.nan,dtype=np.float)
        for i in range(period-1,len(np_array)):
            temp = np.mean(np_array[i-period+1:i+1,:],axis=0)
            res[i,:] = temp
        res = self.TransNPtoOriDF(res,df)
        return res

    def ColumnStd(self,df,period):
        np_array = self.TransDFtoNP(df)
        res = np.full_like(np_array,np.nan,dtype=np.float)
        for i in range(period-1,len(np_array)):
            temp = np.std(np_array[i-period+1:i+1,:],axis=0)
            res[i,:] = temp
        res = self.TransNPtoOriDF(res,df)
        return res

    def ColumnZscore(self,df,period):
        res = (df-self.ColumnMean(df,period))/self.ColumnStd(df,period)
        return res

    def ColumnCorr(self,df,col_list):
        corr_matrix=df[col_list].corr()
        return corr_matrix

    def ColumnCorrDesc(self,df):
        res = self.ColumnCorrAsc(df)
        res.reverse()
        return res

    def ColumnCorrAsc(self,df):
        corr_matrix = self.ColumnCorr(df,list(df.columns)).values
        firms = df.columns
        res = []
        for i in range(1,len(df.columns)):
            for j in range(0,i):
                if corr_matrix[i,j] == corr_matrix[i,j]:
                    d,f = firms[i],firms[j]
                    res.append([(d,f),np.round(corr_matrix[i,j],4)])
        res.sort(key=lambda x: x[1])
        return res

    def ColumnMax(self,df,period):
        np_array = self.TransDFtoNP(df)
        res = np.full_like(np_array,np.nan,dtype=np.float)
        for i in range(period-1,len(np_array)):
            temp = np.max(np_array[i-period+1:i+1,:],axis=0)
            res[i,:] = temp
        res = self.TransNPtoOriDF(res,df)
        return res

    def ColumnMin(self,df,period):
        np_array = self.TransDFtoNP(df)
        res = np.full_like(np_array,np.nan,dtype=np.float)
        for i in range(period-1,len(np_array)):
            temp = np.min(np_array[i-period+1:i+1,:],axis=0)
            res[i,:] = temp
        res = self.TransNPtoOriDF(res,df)
        return res

    def NDayCorrOfTwoArray(self,df1,df2,period): #shape of df1,df2 must be the same
        np1,np2 = self.TransDFtoNP(df1),self.TransDFtoNP(df2)
        res = np.full_like(np1,np.nan,dtype=np.float)
        for i in range(period-1,len(np1)):
            for j in range(len(np1[i])):
                temp = np.corrcoef(np1[i-period+1:i+1,j],np2[i-period+1:i+1,j])[1,0]
                res[i,j] = temp
        res = self.TransNPtoOriDF(res,df1)
        return res

    def Compound(self,df,plus_one=False):
        cal_df = df.copy()
        if plus_one == True:
            cal_df += 1
        cal_df = cal_df.fillna(1)
        res = cal_df.copy()
        for i in range(1,len(df)):
            temp = cal_df.shift(i).fillna(1)
            res *= temp
        return res
        
    def DisCompound(self,df,minus_one=False):
        cal_df = df.copy()
        cal_last = cal_df.shift(1)
        res = cal_df/cal_last
        res = res.fillna(1)
        if minus_one == True:
            res -= 1
        return res
    
    def NDayRetBwd(self,df,period,plus_one=1):
        np_array = self.TransDFtoNP(df)+plus_one
        res = np.full_like(np_array,np.nan,dtype=np.float)
        for i in range(period-1,len(np_array)):
            temp = np.prod(np_array[i-period+1:i+1,:],axis=0)-plus_one
            res[i,:] = temp
        res = self.TransNPtoOriDF(res,df)
        return res

    def NDayRetFwd(self,df,period,plus_one=1):
        np_array = self.TransDFtoNP(df)+plus_one
        res = np.full_like(np_array,np.nan,dtype=np.float)
        for i in range(period-1,len(np_array)):
            temp = np.prod(np_array[i:i+period,:],axis=0)-plus_one
            res[i,:] = temp
        res = self.TransNPtoOriDF(res,df)
        return res

    def Sign(self,df):
        np_array = self.TransDFtoNP(df)
        res = np.sign(np_array)
        res = self.TransNPtoOriDF(res,df)
        return res

    def SignStreak(self,df):
        array = self.TransDFtoNP(df)
        res = np.full_like(array,np.nan)
        res[0,:] = np.sign(array[0,:])
        for j in range(len(array[0])):
            for i in range(1,len(array)):
                last_sign = np.sign(res[i-1,j])
                if array[i,j]*last_sign > 0:
                    res[i,j] = last_sign*(1+abs(res[i-1,j]))
                elif array[i,j]*last_sign < 0:
                    res[i,j] = -last_sign
                elif array[i,j]*last_sign == 0:
                    res[i,j] = res[i-1,j]
        res = self.TransNPtoOriDF(res,df)
        return res
    
    def Rank(self,df,max_has_maxrank=False):
        res = df.rank(axis=1,method='average',ascending=max_has_maxrank)
        return res

    def PR(self,df,max_has_maxrank=True):
        rank = self.Rank(df,max_has_maxrank=max_has_maxrank)
        row_max = rank.max(axis=1)
        div = self.Extend1DTSto2D(row_max,rank)
        res = 100*rank/div
        return res

    def LinearCombination(self,weight_list,df_list):
        linear_combination = 0
        for i in range(0,len(df_list)):
            linear_combination += weight_list[i]*df_list[i]
        return linear_combination
    
    def Perceptron(self,weight_list,df_list,activate_func,params_for_actfunc):
        lc = self.LinearCombination(weight_list,df_list)
        params_for_actfunc['df']=lc
        res = getattr(self,activate_func)(**params_for_actfunc)
        return res

    def StringEqual(self,df,analog_str_list): # df = str in str_list? 1,0
        res = np.full_like(df,0)
        for a_str in analog_str_list:
            res += np.where(df==a_str,1,0)
        res = self.TransNPtoOriDF(res,df)
        return res

    def StringElimination(self,df,analog_str_list): # df = str in str_list? 0,1
        res = np.full_like(df,1,dtype=int)
        for a_str in analog_str_list:
            res *= np.where(df==a_str,0,1)
        res = self.TransNPtoOriDF(res,df)
        return res

    def Extend1DTSto2D(self,df_1d,target_df):
        index1 = df_1d.index
        index2 = target_df.index
        missing_dates = index2.difference(index1)
        res = df_1d.T
        if len(missing_dates) > 0:
            res[missing_dates] = pd.DataFrame([[np.nan for i in range(len(missing_dates))]])
        res = res[index2].values.flatten()
        result = np.tile(res,(target_df.shape[1],1)).T
        result_df = self.TransNPtoOriDF(result,target_df)
        return result_df

    def Extend1DCSto2D(self,df_1d,target_df):
        col1 = df_1d.columns
        col2 = target_df.columns
        missing_firms = col2.difference(col1)
        res = df_1d
        if len(missing_firms) > 0:
            res[missing_firms] = pd.DataFrame([[np.nan for i in range(len(missing_firms))]])
        res = res[col2].values.flatten()
        result = np.tile(res,(target_df.shape[0],1))
        result_df = self.TransNPtoOriDF(result,target_df)
        return result_df

    def MissingColumns(self,ori_df,target_df):
        correct_columns = pd.Index(target_df.columns)
        input_columns = pd.Index(ori_df.columns)
        return list(correct_columns.difference(input_columns))

    def TrimColumns(self,ori_df,target_df):
        lack_tickers = self.MissingColumns(ori_df,target_df)
        for t in lack_tickers:
            ori_df.insert(0,t,np.nan)
        res = ori_df[target_df.columns]
        return res

    def TrimDF(self,ori_df,target_df): #columns must be the same,index must be timestamp or datetime
        ori_index,ori_col = ori_df.index,ori_df.columns
        target_index,target_col = target_df.index,target_df.columns
        if target_col.equals(ori_col) == False:
            print('Columns not match')
            ori_df = self.TrimColumns(ori_df,target_df)

        if target_index.equals(ori_index) == True:
            print('Job already Done')
            return ori_df

        ori_np,target_np = ori_df.values,target_df.values
        result_np = np.full_like(target_np,np.nan)
        current_seat = 0
        ori_first_day = ori_index[0]
        ori_last_day = ori_index[-1]

        for i in range(len(target_index)):
            tar_d = target_index[i]
            if tar_d < ori_first_day:
                pass
                #print(tar_d,'Data not Found')
            elif tar_d == ori_first_day:
                result_np[i,:] = ori_np[0,:]
            elif tar_d > ori_last_day:
                result_np[i,:] = ori_np[-1,:]
            else: #that is, tar_d > ori_first_day
                for k in range(current_seat,len(ori_index)):
                    #print(tar_d,ori_index[k])
                    if tar_d == ori_index[k]:
                        print(tar_d,ori_index[k])
                        current_seat = k-1
                        result_np[i,:] = ori_np[k,:]
                        break  
                    elif tar_d < ori_index[k]:
                        print(tar_d,ori_index[k])
                        current_seat = k-1
                        result_np[i,:] = ori_np[k-1,:]
                        break         
        result_df = self.TransNPtoOriDF(result_np,target_df)
        return result_df

    def MaxOfManyArray(self,candidate):
        res = np.maximum(candidate[0],candidate[1])
        for i in range(2,len(candidate)):
            res = np.maximum(res,candidate[i])
        return res

    def MinOfManyArray(self,candidate):
        res = np.minimum(candidate[0],candidate[1])
        for i in range(2,len(candidate)):
            res = np.minimum(res,candidate[i])
        return res
        
        
class BacktestData(Operators,DrawPlotByTS):
    database={}
    def __init__(self):
        print('Initializing BacktestData...')
        self.path = 'C://StockDB/data/'
        self.cpath = 'C://StockDB/cache/'
        print(self.path)
        self.ImportData('ret')
        self.CheckRet('ret')
        self.ImportData('bm_ret')
        self.CheckRet('bm_ret')
        self.firms = self.database['ret'].columns
        print('Successfully Initializing BacktestData')
    
    def CheckRet(self,ret_df_name):
        check_ret = np.nanmean(abs(self.database[ret_df_name]),axis=1)[1]
        print(ret_df_name,check_ret)
        if check_ret > 0.5:
            decision = input('Unit of '+ret_df_name+' is Wrong, Convert it?(Y/N)')
            if decision == 'Y' or decision == 'y':
                self.database[ret_df_name] *= 0.01
                self.ExportData(ret_df_name)

    def Import(self,df_name,folder_url,encoder):
        file_name = df_name+'.csv'
        datas = os.listdir(folder_url)
        if file_name not in datas:
            print('-Error:',file_name,' Not Found.')
            yes_or_no = input("Build an Empty File to Continue?(Y/N)")
            if yes_or_no == 'Y' or yes_or_no == 'y':
                pd.DataFrame([[0.01],[0.01],[0.01]]).to_csv(folder_url+file_name)
                print('Successfully Build an Empty File to Continue!')
            else:
                print('Import Process Interupted.')
                return 0
        self.database[df_name]=pd.read_csv(folder_url+file_name,index_col=0,parse_dates=True,encoding=encoder)
    
    def ImportData(self,_df_name,_encoder='cp950'):
        self.Import(df_name=_df_name,folder_url=self.path,encoder=_encoder)
    
    def ImportCache(self,_df_name,_encoder='cp950'):
        self.Import(df_name=_df_name,folder_url=self.cpath,encoder=_encoder)
    
    def AddData(self,df,df_name):
        self.database[df_name] = df
    
    def ExportData(self,df_name,encoder='cp950'):
        existing_files = os.listdir(self.cpath)
        file_name = df_name+'.csv'
        if file_name not in existing_files:
            self.database[df_name].to_csv(self.cpath+file_name,encoding=encoder)
        else:
            yes_or_no = 'Y'#input("Want to overwrite existing data?(Y/N)")
            if yes_or_no == 'Y' or yes_or_no == 'y':
                self.database[df_name].to_csv(self.cpath+file_name,encoding=encoder)
            else:
                self.database[df_name].to_csv(self.cpath+df_name+'(2).csv',encoding=encoder)
                
    def ShowDFList(self):
        df_key = list(self.database.keys())
        df_key.sort()
        return df_key

class CalculatePosition(BacktestData,Operators):
    def __init__(self):
        self._reference_df = self.database['ret']
        self.firms = self.database['ret'].columns
    
    def Multiply(self,pos_list): #Logic-'AND'
        res = pos_list[0].copy()
        for i in range(1,len(pos_list)):
            res *= pos_list[i]
        return res
    
    def SumUp(self,pos_list): #Logic-'OR' ,Can also be Used to combine Long/Short Position
        res = pos_list[0].copy()
        for i in range(1,len(pos_list)):
            res += pos_list[i]
        return res
        
    def Weight(self,pos,weight): #weight can be constant,df,np
        res = pos*weight
        return res
    
    def PickByDF(self,pos,pick_df,pick_howmany,descending=True):
        ndarr = self.TransDFtoNP(pos)
        res = np.full_like(ndarr,0)
        for i in range(len(ndarr)):
            exist_pos = list(np.where(ndarr[i]>0)[0])
            if len(exist_pos)>pick_howmany:
                sel = pick_df.iloc[i,exist_pos]
                sel_rank = sel[np.argsort(sel)]
                if descending == True:
                    for j in range(1,pick_howmany+1):
                        res[i,list(self.firms).index(sel_rank.index[-j])] = ndarr[i,list(self.firms).index(sel_rank.index[-j])]
                else:
                    for j in range(pick_howmany):
                        res[i,list(self.firms).index(sel_rank.index[j])] = ndarr[i,list(self.firms).index(sel_rank.index[j])]
            else:
                for j in exist_pos:
                    res[i,j] = ndarr[i,j]
        return res

    def PickByRandom(self,pos,pick_howmany):
        ndarr = self.TransDFtoNP(pos)
        res = np.full_like(ndarr,0)
        for i in range(len(pos)):
            exist_pos = list(np.where(ndarr[i]>0)[0])
            if len(exist_pos)>pick_howmany:
                sel = rdn.sample(exist_pos,pick_howmany)
                for j in sel:
                    res[i,j] = ndarr[i,j]
            else:
                for j in exist_pos:
                    res[i,j] = ndarr[i,j]
        res = self.TransNPtoOriDF(res,pos)
        return res

    def Neutral(self,pos):
        pos_adj = np.nanmean(pos,axis=1)
        pos_adj = pd.DataFrame(pos_adj,index=pos.index)
        pos_adj = self.Extend1DTSto2D(pos_adj,pos)
        res = pos - pos_adj
        return res

    def SegNeutral(self,pos,segment_df):
        segment_np,pos_np = segment_df.values,pos.values
        seglist = set(list(segment_np[0,:]))
        res = np.full_like(pos,0,dtype=np.float)
        for seg in seglist:
            seg_pos = np.where(segment_np==seg,pos_np,np.nan)
            seg_pos = self.TransNPtoOriDF(seg_pos,pos)
            seg_pos_demean = self.Neutral(seg_pos)
            seg_pos_demean_masked = np.where(segment_np==seg,seg_pos_demean,0)
            res += seg_pos_demean_masked 
        res = self.TransNPtoOriDF(res,pos)
        return res

    def Scale(self,pos):
        pos_adj = np.sum(abs(pos),axis = 1)
        for i in range(len(pos_adj)):
            if pos_adj[i] == 0:
                pos_adj[i] = 1
        pos_adj = np.tile(pos_adj,(len(self.firms),1)).T
        res = pos/pos_adj
        return res
    
    def Timing(self,pos,time_filter):
        res = pos * time_filter
        return res
    
    def AdjustFreq(self,pos,start,new_freq):
        res = self.TransDFtoNP(pos)
        for i in range(start,len(res),new_freq):
            for ii in range(1,new_freq):
                if i+ii < len(res):
                    res[i+ii,] = res[i,]
        return res

    def MaxHoldingCap(self,pos,cap):
        res = self.TransDFtoNP(pos)
        res = np.where(res>cap,cap,res)
        res = self.TransNPtoOriDF(res,pos)
        return res

    def MinHoldingPeriod(self,pos,period): #Use this func before scaling and weighting!
        #print('Hint: Input boolin array into MinHoldingPeriod!')
        res = self.TransDFtoNP(pos)
        for j in range(1,len(res[0])):
            signals = [] #找出第一次出現訊號的位置
            for i in range(len(res)):
                if res[i,j] != 0 and res[i-1,j] == 0:
                    signals.append(i)
            for s in signals:
                for p in range(1,period):
                    if s+p<len(res):
                        res[s+p,j] = res[s,j]
        res = self.TransNPtoOriDF(res,pos)
        return res

    def MinHoldingPeriodConditional(self,pos,period,condi): #Use this func before scaling and weighting! Condi is a boolin array.在period期間，符合condi就繼續持有
        #print('Hint: Input boolin array into MinHoldingPeriod!')
        res = self.TransDFtoNP(pos)
        for j in range(1,len(res[0])):
            signals = [] #找出第一次出現訊號的位置
            for i in range(len(res)):
                if res[i,j] != 0 and res[i-1,j] == 0:
                    signals.append(i)
            for s in signals:
                for p in range(1,period):
                    if s+p<len(res):
                        if condi[s+p,j] == 1:
                            res[s+p,j] = res[s,j]
                        else:
                            break
        res = self.TransNPtoOriDF(res,pos)
        return res


    def ExPosition(self,pos):
        pos_condi_last = np.roll(pos,1,axis=0)
        pos_condi_last[0,:] = 0
        pos_condi_ex = self.BinaryStep(pos,(0,0))*self.BinaryStep(pos_condi_last,(1,1))
        return pos_condi_ex

    def OutputDF(self,pos):
        res = self.TransNPtoOriDF(np_array=pos,ori_matrix=self._reference_df)
        return res
    
class Backtest(BacktestData,Operators,DrawPlotByTS):
    def __init__(self,pos,start_date,end_date,extend_date,valid_period,fee_rate,period_in_year,alternative,fast=False): #valid_period:Some Strategy(e.g.MA) needs time to prepare data
        self.start_date = start_date
        self.end_date = end_date
        self.extend_date = extend_date
        self.pos_str_df = pos
        self.valid_period = valid_period
        self.SetDataUsed()
        self.fee_rate = fee_rate
        self.piy = period_in_year
        self.alternative = alternative
        self.Go(fast)
        self.firms = self.able_position.columns
        self.dates = self.able_position.index
    
    def SetDataUsed(self):
        self.able_position = self.pos_str_df.loc[self.start_date:self.end_date,]
        self.able_bm_return = self.database['bm_ret'].loc[self.start_date:self.end_date,]
        self.able_return = self.database['ret'].loc[self.start_date:self.end_date,]

        self.valid_date = self.able_position.index[self.valid_period]
        self.able_position = self.able_position.iloc[self.valid_period:,]
        self.able_bm_return = self.able_bm_return.iloc[self.valid_period:,]
        self.able_return = self.able_return.iloc[self.valid_period:,]

        self.able_dates = self.database['ret'].loc[self.valid_date:self.extend_date].index
        #print(self.able_bm_return.shape,self.able_position.shape,self.able_return.shape,self.able_dates.shape)
    
    def HowToUseIdleCash(self,asset_name):
        if asset_name == None or asset_name == 'Cash' or asset_name == 'cash':
            return 0
        else:
            time = self.able_return.index
            asset_ret = self.database[asset_name]
            return asset_ret.loc[time].values.flatten()
    
    def Winrate(self):
        wins = 0
        abs_wins = 0
        for i in range(len(self.single_bm)):
            if self.single_str.iloc[i,0] >= self.single_bm.iloc[i,0]:
                wins += 1
            if self.single_str.iloc[i,0] > 1:
                abs_wins += 1
        winrate = wins/len(self.single_bm)
        print('Winrate:','%s'% (np.round(100*winrate,2))+'%')
        abs_winrate = abs_wins/len(self.single_bm)
        print('ABS_Winrate:','%s'% (np.round(100*abs_winrate,2))+'%')

    def Turnover(self):
        res = print('Turnover Rate:','%s'% (np.round(100*np.mean(self.dif_position_rowsum),2))+'%')
    
    def CalculateReturn(self):
        #print(self.alternative)
        self.alternative_ret = self.HowToUseIdleCash(self.alternative)
        self.daily_return = pd.DataFrame.sum(self.able_position * self.able_return,axis = 1,skipna=True) + (1-pd.DataFrame.sum(self.able_position,axis=1))*self.alternative_ret
        self.daily_return = pd.DataFrame(self.daily_return,columns=['sum'])
        self.dif_position = self.BwdChangeNumber(self.able_position)
        self.dif_position = self.FillRowN(self.dif_position,fill_in_content=self.able_position,row_number=0)
        self.dif_position_rowsum = abs(self.dif_position).sum(axis=1,skipna=True)
        self.change_loss = pd.DataFrame(self.fee_rate * self.dif_position_rowsum,columns=['sum'])
        #self.change_loss = pd.DataFrame((self.fee_rate * abs(self.dif_position)).sum(axis=1,skipna=True),columns=['sum'])
        #print(change_loss.shape,daily_return.shape)
        
    def Compounding(self):
        self.single_str = 1 + self.daily_return - self.change_loss
        self.single_bm = 1 + self.able_bm_return
        self.path_str = self.Compound(self.single_str,False)
        self.path_str.index = self.able_dates[1:]
        self.path_bm = self.Compound(self.single_bm,False)
        self.path_bm.index = self.able_dates[1:]
        #print(len(self.path_str),len(self.path_bm))
        
    def ExportPlot(self):
        self.PlotLogPicture(["allocation","benchmark"],self.piy,self.single_str['sum'],self.single_bm['bm']) #label,1年有幾期,單一日報酬df
        self.PlotDrawdown(["allocation","benchmark"],self.path_str['sum'],self.path_bm['bm']) #label,累積日報酬df
        
    def GetBasicInf(self):
        self.Winrate()
        self.Turnover()
        self.cbi = self.CalculateBasicInformation(self.path_str['sum'],self.single_str['sum'],self.piy)
        self.gbi_dict = {'Ret':self.cbi[0],'Std':self.cbi[1],'Sharpe':self.cbi[2],'DD':self.cbi[3],'RtoDD':self.cbi[4]}
        return self.gbi_dict

    def GetBasicInf_fast(self):
        self.cbi = self.CalculateBasicInformationFast(self.path_str['sum'],self.single_str['sum'],self.piy)
        self.gbi_dict = {'Ret':self.cbi[0],'Std':self.cbi[1],'Sharpe':self.cbi[2],'DD':self.cbi[3],'RtoDD':self.cbi[4]}
        return self.gbi_dict
    
    def Go(self,fastmode):
        if fastmode == False:
            self.CalculateReturn()
            self.Compounding()
            self.finish = self.GetBasicInf()
            print(self.finish)
        elif fastmode == True:
            self.CalculateReturn()
            self.Compounding()
            self.finish = self.GetBasicInf_fast()
            #print(np.round(self.finish['Sharpe'],4))
        
    def RetDistribution(self,bins=25):
        strategy_ret = self.single_str.values
        plt.hist(strategy_ret,bins)
        plt.show()

    def NumberOfHolding(self):
        pos = self.able_position
        long_or_short = np.where((pos>0)|(pos<0),1,0)
        res = np.sum(long_or_short,axis=1)
        res = pd.DataFrame(res,index=pos.index)
        return res

    def OutplayGraph(self):
        strategy_ret,bm_ret = self.path_str.values,self.path_bm.values
        div = pd.DataFrame((strategy_ret/bm_ret),index=self.path_str.index)
        plt.figure(figsize=(15,3))
        plt.plot(div)
        plt.yscale("log")
        plt.show()

    def LargestPosition(self):
        pos,firm_ret = self.able_position,self.able_return
        most = np.max(pos,axis=1)
        firms,dates = pos.columns,pos.index
        position_np = pos.values
        firm_ret_np = strategy_ret.values
        for i in range(len(most)):
            for j in range(len(firms)):
                if position_np[i,j] == most[i]:
                    print(dates[i],firms[j],most[i],'/ Its return:',firm_ret_np[i,j])
                    break

    def Receipt(self,start_date=0,end_date=0):
        pos = self.able_position
        pos_np,ret_np = self.TransDFtoNP(pos),self.able_return.values
        pos_index,pos_columns = pos.index,pos.columns
        if end_date == 0:
            end_date = len(pos_np)
        for i in range(start_date,end_date):
            print(pos_index[i])
            for j in range(len(pos_np[i])):
                if pos_np[i,j] > 0:
                    print(pos_columns[j],np.round(ret_np[i,j],4))

    def WeekdayPerformance(self):
        res = [[],[],[],[],[]]
        for i in range(len(self.single_str)):
            wd = dt.datetime.weekday(self.single_str.index[i])
            ret = self.single_str.values[i][0]
            try:
                res[wd].append(ret)
            except:
                continue
        res = np.asarray(res)
        data = {'Mon':np.mean(res[0])-1,'Tue':np.mean(res[1])-1,'Wed':np.mean(res[2])-1,'Thu':np.mean(res[3])-1,'Fri':np.mean(res[4])-1}
        return data

    def ShowInfoOfSpecifiedStocks(self,condition,*database): #condition should be a boolin matrix that represents what you care
        pos = self.pos_str_df
        trimmed_data = []
        for db in database:
            temp = self.TrimDF(ori_df=db,target_df=pos)
            trimmed_data.append(temp.values)
        position_np = pos.values()
        firms,dates = pos.columns,pos.index
        for i in range(len(dates)):
            for j in range(len(firms)):
                if condition[i,j] == 1:
                    print(dates[i],firms[j])
                    for td in trimmed_data:
                        print(td,td[i,j])

    def RollingTest(self,interval_length):
        strategy_ret = self.single_str['sum']
        bm_ret = self.single_bm['bm']
        period_in_year = self.piy

        for i in range(0,len(strategy_ret),interval_length):
            print('Period',i)
            switch = 0
            for ret in [strategy_ret,bm_ret]:
                if switch == 0:
                    switch += 1
                    print('Str:')
                else:
                    print('Benchmark:')
                interval_ret = ret.iloc[i:i+interval_length]
                interval_path = self.Compound(interval_ret,False)
                cbi = self.CalculateBasicInformation(interval_path,interval_ret,period_in_year)
                print(cbi)

    def StatOfEachBet(self):
        pos,ret = self.able_position.values,self.able_return.values
        record = []
        for i in range(len(pos)):
            for j in range(len(pos[i])):
                if pos[i,j] > 0:
                    record.append(ret[i,j])
        tstat = (m.sqrt(len(record))*np.nanmean(record)/(np.nanstd(record)))
        res = (np.nanmean(record),np.nanstd(record),len(record),tstat)
        #print(res)
        return res

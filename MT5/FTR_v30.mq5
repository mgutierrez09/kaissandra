//+------------------------------------------------------------------+
//|                                                      FTR_v30.mq5 |
//|                                                       mgutierrez |
//|                                       https://www.kaissandra.com |
//|        Differentiation between strategies within the same symbol |
//+------------------------------------------------------------------+
#property copyright "mgutierrez"
#property link      "https://www.kaissandra.com"
#property version   "3.00"
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\OrderInfo.mqh>

#define EXPERT_MAGIC 123456   // MagicNumber of the expert

// Define variables
int const nBuffers = 100;
int const bufferSize = 10;
double buffer[10][2];// buffer containing last bids/asks
int file_index;
int bids_counter;
int filehandle;

datetime currTime;
datetime timeSoll;
datetime tic;
datetime toc;
double diffTime;
double bid;
double ask;
double Bis[];
double Ais[];
double Bi_solls[];
double Ai_solls[];
double real_profit;

string filename;
string logfilename;
string thisSymbol;
string directoryNameLive;
string directoryNameLog;
string directoryNameComm;
string directoryNameRecordings;
string directoryNameAccount;
string chunks[];
string posinfo[];
int thisPositionsTotal;

// Define variables
CTrade m_Trade;
CPositionInfo m_Position;
//input double Lot = 0.01;
double lot = 0.01;
const double lot_in_eur = 100000;
bool record_data = true;
bool closingInProgress;

int filehandle_trader;
int filehandlelog;
long nEventsPerStats[];
string predictionString;
long predictions[];
string TTfile;
string LCfile;
bool timer_bool;
string close_type="";
//datetime currTime;

double stoplosses[];
double takeprofits[];
double sl_protects[];
double tp_protects[];
double sl_protect_prevs[];
double tp_protect_prevs[];
double sl_thr = 1;//0.0001; //in ratio (1 pip=0.0001)
double slThrPips = 100;
double tpThrPips = 1000;
double slProThrPips = 1000;
double const PIP = 0.0001;
//double bid;
//double ask;
double inter_profit;
double profit;
double GROI;
double ROI;
double spread;
//int position = 0; // type of position opened: 1 long/-1 short
int positions[]; // type of position opened: 1 long/-1 short
long deadlines[];
ulong pos_tickets[];
//long deadline = -1;
int numStragtegies;
int ticks_counter = 0;

string      stringThisDate;
datetime    thisTime;
MqlDateTime timeStruct;
MqlDateTime currentTimeStruct;
//string      filename;
int         filehandleTest;
int         filehandle_record;

int first_pos = 0;
int ticks_counter_open = 0;
long difs_ticks[];
int n_pos_open = 0;
int sit = -1; // strategy index in transition for closing
int sot = -1; // strategy index in transition for opening
//int firstBuff[2] = {1,1};

// Strategy variables


void openPosition(string origin, int thisPos, int str_idx){
   // wait for other positions to open if its the case
   while(sot!=-1){
      Print("WARNING! Open position called but sot!=-1! Waiting.");
      Sleep(1000);
      }
   Bis[str_idx] = bid;
   Ais[str_idx] = ask;
   positions[str_idx] = thisPos;
   string message;
   //int n_pos = PositionsTotal();
   if(thisPos==1){
      while(!m_Trade.Buy(lot,thisSymbol)){
         //string message = StringFormat();
         message = StringFormat("WARNING! Buy -> false. Result Retcode: %u, description of result: %s",m_Trade.ResultRetcode(),m_Trade.ResultRetcodeDescription());
         Print(message);
         //writeLog(message);
         //Print("WARNING! Buy -> false. Result Retcode: ",m_Trade.ResultRetcode(),", description of result: ",m_Trade.ResultRetcodeDescription());
      }
      thisPositionsTotal ++;
      sot = str_idx;
      pos_tickets[str_idx] = 0;//PositionGetInteger(POSITION_IDENTIFIER);
      //string message = StringFormat("Buy -> talse. Result Retcode: ",m_Trade.ResultRetcode(),", description of result: ",m_Trade.ResultRetcodeDescription());
      message = StringFormat("Buy -> true. Result Retcode: %u, description of result: %s",m_Trade.ResultRetcode(),m_Trade.ResultRetcodeDescription());
      //Print("Buy -> true. Result Retcode: ",m_Trade.ResultRetcode(),", description of result: ",m_Trade.ResultRetcodeDescription());
      Print(message);
      
      // Save timestap
         
      positions[str_idx] = thisPos;
      //writeLog(message);
      //stoploss = -Ai*sl_thr+Bi;
      stoplosses[str_idx] = updateSL(Ais[str_idx], thisPos, slThrPips);
      takeprofits[str_idx] = updateTP(Ais[str_idx], thisPos, tpThrPips);
   }
   else{if(thisPos==-1){
      while(!m_Trade.Sell(lot,thisSymbol)){
         //string message = StringFormat();
         message = StringFormat("WARNING! Sell -> false. Result Retcode: %u, description of result: %s",m_Trade.ResultRetcode(),m_Trade.ResultRetcodeDescription());
         Print(message);
        // writeLog(message);
         //Print("WARNING! Sell -> false. Result Retcode: ",m_Trade.ResultRetcode(),", description of result: ",m_Trade.ResultRetcodeDescription());
      }
      thisPositionsTotal ++;
      sot = str_idx;
      pos_tickets[str_idx] = 0;//PositionGetInteger(POSITION_IDENTIFIER);
      message = StringFormat("Sell -> true. Result Retcode: %u, description of result: %s",m_Trade.ResultRetcode(),m_Trade.ResultRetcodeDescription());
      Print(message);
      
      // Save timestap
         
      positions[str_idx] = thisPos;
      //writeLog(message);
      //Print("Sell -> true. Result Retcode: ",m_Trade.ResultRetcode(),", description of result: ",m_Trade.ResultRetcodeDescription());
      //stoploss = Bi*sl_thr+Ai;
      stoplosses[str_idx] = updateSL(Bis[str_idx], thisPos, slThrPips);
      takeprofits[str_idx] = updateTP(Bis[str_idx], thisPos, tpThrPips);
   }
      else{
         Print("ERROR! Position cannot be zero");
      }
   }
   //Print("m_Trade.ResultRetcode()");
   //if(m_Trade.ResultRetcode()==TRADE_RETCODE_DONE)
   //   Print(m_Trade.ResultRetcode());
   //else
   //   Print("ResultRetcode() wrong");
   //Print("m_Trade.ResultRetcodeDescription()");
   //Print(m_Trade.ResultRetcodeDescription());
   //Print("m_Trade.ResultDeal()");
   //Print(m_Trade.ResultDeal());
   toc = TimeCurrent();
   deadlines[str_idx] = 1; // reset deadline
   Bi_solls[str_idx] = buffer[bufferSize-1][0];
   Ai_solls[str_idx] = buffer[bufferSize-1][1];
   difs_ticks[str_idx] = ticks_counter-ticks_counter_open;
   message = StringFormat("%d. #Events %d Ticks %d BiS %.4f BiI %.4f AiS %.4f AiI %.4f SL %.4f SP %.4f",thisPos,nEventsPerStats[str_idx],difs_ticks[str_idx],Bi_solls[str_idx], 
         Bis[str_idx],Ai_solls[str_idx],Ais[str_idx],stoplosses[str_idx],(Ais[str_idx]-Bis[str_idx])/Ais[str_idx]);
   Print(message);
}


void closePosition(int str_idx){
   if(positions[str_idx]!=0){
      int executed = 0;
      //bool closed = false;
      string message;
      int counter = 0;
      //while(!executed && position!=0){
      while(!m_Trade.PositionClose(pos_tickets[str_idx]) && counter<5){
         message = StringFormat("WARNING! Close position executed: %d",executed);
         Print(message);
         // Send Warning message to trader that execution wasn't successful
         int fh = FileOpen(directoryNameComm+"WARNING",FILE_WRITE|FILE_CSV|FILE_ANSI,',');
         if(fh>0){
            FileWrite(fh,"POSITION NOT CLOSED");
            FileClose(fh);
         }
         Sleep(1000);
         counter ++;
      }
      if(counter<5)executed = 1;
      message = StringFormat("Close position executed: %d",executed);
      Print(message);
   
      message = StringFormat("Close -> true. Result Retcode: %u, description of result: %s",m_Trade.ResultRetcode(),m_Trade.ResultRetcodeDescription());
      Print(message);
      //writeLog(message);
      
      closingInProgress = true;
      thisPositionsTotal --;
      
      
      if (sit!=-1){
         Print("WARNING! Overlap of transition of closing positions!");
      }
      sit = str_idx;
      //}
      /*else{
         message = StringFormat("WARNING! Close position executed: %d",executed);
         Print(message);
         // Send Warning message to trader that execution wasn't successful
         int fh = FileOpen(directoryNameComm+"WARNING",FILE_WRITE|FILE_CSV|FILE_ANSI,',');
         if(fh>0){
            FileWrite(fh,"POSITION NOT CLOSED");
            FileClose(fh);
         }
      }*/
   }
   else{
      string message = "WARNING! Try to close position but no position is open. Skipped";
      Print(message);
      // Send CL in case it's unsynched with trader
      //filename = "CL";
      //filehandleTest = FileOpen(directoryNameLive+filename,FILE_WRITE|FILE_CSV|FILE_ANSI,',');
      //FileWrite(filehandleTest,thisSymbol,toc,TimeCurrent(),position,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,0);
      //FileClose(filehandleTest);
      //writeLog(message);
   }

}
//+------------------------------------------------------------------+
//| Check for opening                                                |
//+------------------------------------------------------------------+
void checkForOpening(int s){
//---
   // Launch position 
   string message;
   string ttfilename = StringFormat("%s%s%d",directoryNameLive,TTfile,s);
   if(FileIsExist(ttfilename)==1 && !closingInProgress){
      message = "TT found";
      Print(message);
      //writeLog(message);
      // open position
         first_pos = 1;
         
         
         // Read type of trigger and delete TT
         int k;
         bool reset = false;
         int tries = 0;
         int str_idx =-1;
         do{
            filehandle_trader = FileOpen(ttfilename,FILE_READ|FILE_ANSI);
            predictionString = FileReadString(filehandle_trader);
            k = StringSplit(predictionString,StringGetCharacter(",",0),chunks);
            FileClose(filehandle_trader);
            tries = tries+1;
         }while(k!=5);
         str_idx = (int)StringToInteger(chunks[4]);
         message = StringFormat("Strategy index %d",str_idx);
         Print(message);
         predictions[str_idx] = StringToInteger(chunks[0]);
         lot = StringToDouble(chunks[1]);
         slThrPips = (int)StringToInteger(chunks[3]);
         
         //Print("slThrPips %d",slThrPips);
         // extend deadline only when new deadline is further in time
         //if(nEventsPerStat-deadline<StringToDouble(chunks[2])){
         nEventsPerStats[str_idx] = StringToInteger(chunks[2]);
         reset = true;
         //}
         
         FileDelete(ttfilename);
         long thr_pred = 0;
         //diffTime = toc-tic;
         // launch long positon
         //Print(prediction);
         if(predictions[str_idx]>thr_pred){
            
            //if the position for this symbol already exists -> extend deadline
            //if(m_Position.Select(thisSymbol)){
               
               //if(m_Position.PositionType()==POSITION_TYPE_SELL) m_Trade.PositionClose(my_symbol);  //and this is a Sell position, then close it
               //if(m_Position.PositionType()==POSITION_TYPE_BUY){
            if(pos_tickets[str_idx]!=-1){
               if(reset){
                  message = StringFormat("Deadline extended by %d",nEventsPerStats[str_idx]);
                  Print(message);
                  // save position state
                  //int fh = FileOpen(directoryNameComm+"POSSTATE.txt",FILE_WRITE|FILE_CSV|FILE_ANSI,',');
                  //FileWrite(fh,positions[str_idx],deadlines[str_idx],Bis[str_idx],Ais[str_idx],difs_ticks[str_idx],nEventsPerStats[str_idx],stoplosses[str_idx],takeprofits[str_idx],lot);
                  //FileClose(fh);
                  // reset deadline
                  deadlines[str_idx] = 1;
               }else{
                  message = StringFormat("Deadline not extended. Remaining %d",nEventsPerStats[str_idx]-difs_ticks[str_idx]-deadlines[str_idx]);
                  Print(message);
                  //writeLog(message);
               }
               
            //}
            /*else{
               openPosition("TICK", 1, str_idx);
               pos_tickets[str_idx] = PositionGetTicket(PositionsTotal());
               n_pos_open = n_pos_open+1;
            }*/
               //}
            }
            else{ // open new long position
               openPosition("TICK", 1, str_idx);
               //pos_tickets[str_idx] = PositionGetTicket(n_pos_open);
               n_pos_open = n_pos_open+1;
            }
            
         }
         
         // launch short positon
         else{if(predictions[str_idx]<thr_pred){
            //if the position for this symbol already exists -> extend deadline
            //if(m_Position.Select(thisSymbol)){
               
               //if(m_Position.PositionType()==POSITION_TYPE_SELL){ 
            if(pos_tickets[str_idx]!=-1){
               if(reset){
                  //dif_ticks = dif_ticks+ticks_counter-ticks_counter_open;
                  message = StringFormat("Deadline extended by %d",nEventsPerStats[str_idx]);
                  Print(message);
                  // save position state
                  //int fh = FileOpen(directoryNameComm+"POSSTATE.txt",FILE_WRITE|FILE_CSV|FILE_ANSI,',');
                  //FileWrite(fh,positions[str_idx],deadlines[str_idx],Bis[str_idx],Ais[str_idx],difs_ticks[str_idx],nEventsPerStats[str_idx],stoplosses[str_idx],takeprofits[str_idx],lot);
                  //FileClose(fh);
                  // reset deadline
                  deadlines[str_idx] = 1;
               }else{
                  message = StringFormat("Deadline not extended. Remaining %d",nEventsPerStats[str_idx]-difs_ticks[str_idx]-deadlines[str_idx]);
                  Print(message);
                  //writeLog(message);
               }
            }
         
            else{
               openPosition("TICK", -1, str_idx);
               //pos_tickets[str_idx] = PositionGetTicket(n_pos_open);
               n_pos_open = n_pos_open+1;
            }
               //}
            //}
            // open new position
            /*else{
               
               openPosition("TICK", -1, str_idx);
               pos_tickets[str_idx] = PositionGetTicket(n_pos_open);
               n_pos_open = n_pos_open+1;
            }*/
            
            // if else
         }else{
               message = StringFormat("ERROR!! Prediction cannot be zero!. Prediction string %s. Tries %d. Prediction %d",predictionString,tries,predictions[str_idx]);
               Print(message);
               //writeLog(message);
         }
         }
      }
      else{
         if(FileIsExist(directoryNameLive+TTfile)==1){
            message = "WARNING! TT found but closing in progress. Opening cancelled and trying to close again.";
            Print(message);
            FileDelete(directoryNameLive+TTfile);
            
            //writeLog(message);
         }
      
      }
   
}

void controlPositionFlow(){
   // Check state of open position
   string message;
   for(int s=0; s<numStragtegies; s++){
      if(deadlines[s]>0){
         if (positions[s]==1){
            GROI = 100*(ask-Ais[s])/Ais[s];
            ROI = 100*(bid-Ais[s])/Ais[s];
            sl_protects[s] = updateSL(ask, positions[s], slProThrPips);
            tp_protects[s] = updateTP(ask, positions[s], slProThrPips);
         }else{if(positions[s]==-1){
            GROI = 100*(Bis[s]-bid)/ask;
            ROI = 100*(Bis[s]-ask)/ask;
            sl_protects[s] = updateSL(bid, positions[s], slProThrPips);
            tp_protects[s] = updateTP(bid, positions[s], slProThrPips);
         }}
         //Print("",nEventsPerStat-dif_ticks-deadline,"",GROI," ",ROI,"%");
         message = StringFormat("Deadline in  %d GROI %.4f ROI %.4f",nEventsPerStats[s]-difs_ticks[s]-deadlines[s],GROI,ROI);
         //stoploss = updateSL(Bi, thisPos, slThrPips);
         //Print(message);
         //writeLog(message);
         
         // SL / TP protections not supported for now
         //if(sl_protects[s]!=sl_protect_prevs[s] || tp_protects[s]!=tp_protect_prevs[s])
         //   m_Trades[s].PositionModify(thisSymbol,sl_protects[s],tp_protects[s]);
         //sl_protect_prevs[s] = sl_protects[s];
         //tp_protect_prevs[s] = tp_protect[s];
         
         // update deadline
         deadlines[s] = (deadlines[s]+1)%(nEventsPerStats[s]-difs_ticks[s]);
         
         // close position if deadline reached
         if (deadlines[s]==0){
            message = "Deadline reached";
            Print(message);
            //writeLog(message);
            close_type = "CL";
            closePosition(s);
         }
         
         else{
            // check if stoploss reached
            if((positions[s]==1 && ask<stoplosses[s]) || (positions[s]==-1 && bid>stoplosses[s])){
               message = "SL reached";
               Print(message);
               //writeLog(message);
               close_type = "SL";
               closePosition(s);
            }
            // check if takeprofit reached
            if((positions[s]==1 && ask>takeprofits[s]) || (positions[s]==-1 && bid<takeprofits[s])){
               message = "TP reached";
               Print(message);
               //writeLog(message);
               // WARNING! Temporal close_type as SL. TP not implemented yet!!!
               close_type = "TP";
               closePosition(s);
            }
         }
         
         //accountInfo();
      }
   }
}

void recordData(){
   if(TimeToStruct(currTime,currentTimeStruct)==false){
         PrintFormat("TimeToStruct falied. Error code %d",GetLastError());
      }
      
      // change of days
      if(currentTimeStruct.day!=timeStruct.day){
         
         // close previous file
         FileClose(filehandle_record); 
         // create new file
         //thisTime = TimeCurrent();
         //ResetLastError(); 
         // Generate time structure
         if(TimeToStruct(currTime,timeStruct)==false){
            PrintFormat("TimeToStruct falied. Error code %d",GetLastError());
         }
         // Generate file name
         stringThisDate = StringFormat("%4d%02d%02d%02d%02d%02d",timeStruct.year,timeStruct.mon,timeStruct.day,timeStruct.hour,timeStruct.min,timeStruct.sec);
         filename = thisSymbol+"_"+stringThisDate+".txt";
      
         filehandle_record=FileOpen(directoryNameRecordings+"//"+filename,FILE_WRITE|FILE_CSV|FILE_ANSI,',');
         FileWrite(filehandle_record,"DateTime","SymbolBid","SymbolAsk");
      }
      
      //FileWrite(filehandle,TimeCurrent(),bid,ask);
      //Print(filehandle);
      
      FileWrite(filehandle_record,currTime,bid,ask);

}

double updateSL(double symbol, int direction, double thrPip){
   double thr = thrPip*PIP;
   return symbol*(1-direction*thr);
}

double updateTP(double symbol, int direction, double thrPip){
   double thr = thrPip*PIP;
   return symbol*(1+direction*thr);
}

void writeLog(string message){
   // Generate time structure
   if(TimeToStruct(TimeCurrent(),timeStruct)==false){
      PrintFormat("TimeToStruct falied. Error code %d",GetLastError());
   }
   string DTstamp = StringFormat("%4d.%02d.%02d %02d:%02d:%02d",timeStruct.year,timeStruct.mon,timeStruct.day,timeStruct.hour,timeStruct.min,timeStruct.sec);
   FileWrite(filehandlelog, DTstamp+" "+thisSymbol+" "+message);
   
}

int OnInit()
  {
//---
   /*int i=PositionsTotal()-1;
   if(Symbol()=="EURUSD"){
   Print("Pos total ",i);
   while (i>=0)
     {
      if (m_Trade.PositionClose(PositionGetSymbol(i))) i--;
     }}*/
   // init arrays
   // TODO: Resize array once we know how many strategies are there
   // TEMP: Fix positions size to two
   numStragtegies = 4;
   ArrayResize(positions, numStragtegies);
   ArrayResize(deadlines, numStragtegies);
   ArrayResize(sl_protects, numStragtegies);
   ArrayResize(tp_protects, numStragtegies);
   ArrayResize(sl_protect_prevs, numStragtegies);
   ArrayResize(tp_protect_prevs, numStragtegies);
   ArrayResize(difs_ticks, numStragtegies);
   ArrayResize(nEventsPerStats, numStragtegies);
   //ArrayResize(m_Trades, numStragtegies);
   //ArrayResize(m_Positions, numStragtegies);
   ArrayResize(stoplosses, numStragtegies);
   ArrayResize(takeprofits, numStragtegies);
   ArrayResize(Bi_solls, numStragtegies);
   ArrayResize(Ai_solls, numStragtegies);
   ArrayResize(Bis, numStragtegies);
   ArrayResize(Ais, numStragtegies);
   ArrayResize(pos_tickets, numStragtegies);
   ArrayResize(predictions, numStragtegies);
   //ArrayResize(closingInProgresses, numStragtegies);
   
   ArrayInitialize(positions, 0);
   ArrayInitialize(deadlines, -1);
   ArrayInitialize(pos_tickets, -1);
   //ArrayInitialize(closingInProgresses, false);
   //ArrayInitialize(difs_ticks, 0);
   file_index = 0;
   thisSymbol = Symbol();
   directoryNameLive = "IOlive//"+thisSymbol+"//";
   directoryNameLog = "Log//"+thisSymbol+"//";
   directoryNameComm = "COMM//"+thisSymbol+"//";
   directoryNameAccount = "Account//";
   directoryNameRecordings = "Data//"+thisSymbol+"//";
   //Print(thisSymbol+" fetcher launched");
   // init bid and ask to avoud division by zero
   bid = 1.0;
   ask = 1.0;
   ulong m_slippage = 100;                // slippage
   thisPositionsTotal = 0;
   
   timer_bool = EventSetMillisecondTimer(100);
   m_Trade.SetDeviationInPoints(m_slippage);

   TTfile = "TT";// trade trigger file signalling
   LCfile = "LC";
   // set milisecond timer
   //timer_bool = EventSetMillisecondTimer(10);
   Print(thisSymbol+" Fetcher-Trader-Recorder version 3.0 launched");
   
   // Test to check open/close instants and synchronizity with Python
   
   // variables
   thisTime = TimeCurrent(); // get Init recording time
   ResetLastError(); 
   // Generate time structure
   if(TimeToStruct(thisTime,timeStruct)==false){
      PrintFormat("TimeToStruct falied. Error code %d",GetLastError());
   }

   // Generate file name
   stringThisDate = StringFormat("%4d%02d%02d%02d%02d%02d",timeStruct.year,timeStruct.mon,timeStruct.day,timeStruct.hour,timeStruct.min,timeStruct.sec);
   filename = thisSymbol+"_"+stringThisDate+".txt";
   
   filehandle_record=FileOpen(directoryNameLive+"//0RESET",FILE_WRITE|FILE_CSV|FILE_ANSI,',');
   FileClose(filehandle_record); 
   //FileWrite(filehandle_record,"DateTime","SymbolBid","SymbolAsk");
   
   filehandle_record=FileOpen(directoryNameRecordings+"//"+filename,FILE_WRITE|FILE_CSV|FILE_ANSI,',');
   FileWrite(filehandle_record,"DateTime","SymbolBid","SymbolAsk");
   // init log file
   logfilename = thisSymbol+"_"+stringThisDate+".log";
   filehandlelog = FileOpen(directoryNameLog+logfilename,FILE_WRITE|FILE_CSV|FILE_ANSI);
   
   logfilename = thisSymbol+"_"+stringThisDate+".log";
   filehandlelog = FileOpen(directoryNameLog+logfilename,FILE_WRITE|FILE_CSV|FILE_ANSI);
   
   string statusfilename;
   for(int s=0; s<numStragtegies; s++){
      statusfilename = StringFormat("%sPOSSTATE%d.txt",directoryNameComm,s);
   // check if position was left open
      if(FileIsExist(statusfilename)==1 ){
         int fh = FileOpen(statusfilename,FILE_READ|FILE_ANSI);
         string info = FileReadString(fh);
         Print(info);
         int k = StringSplit(info,StringGetCharacter(",",0),posinfo);
         FileClose(fh);
         
         positions[s] = (int)StringToInteger(posinfo[0]);
         deadlines[s] = StringToInteger(posinfo[1]);
         Bis[s] = StringToDouble(posinfo[2]);
         Ais[s] = StringToDouble(posinfo[3]);
         difs_ticks[s] = StringToInteger(posinfo[4]);
         nEventsPerStats[s] = StringToInteger(posinfo[5]);
         stoplosses[s] = StringToDouble(posinfo[6]);
         takeprofits[s] = StringToDouble(posinfo[7]);
         pos_tickets[s] = StringToInteger(posinfo[8]);
         lot = StringToDouble(posinfo[9]);
         
         while(!FileDelete(statusfilename));
      }
   }
   saveAccountInfo();
   
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   Print(thisSymbol+" fetcher closed");
   FileClose(filehandle);
   saveAccountInfo();
   FileClose(filehandlelog);
   
   if(FileIsExist(directoryNameComm+"POSINFO.txt")==1 ){
      while(!FileDelete(directoryNameComm+"POSINFO.txt"));
    }
   
   if(FileIsExist(directoryNameComm+"WARNING")==1 ){
      while(!FileDelete(directoryNameComm+"WARNING"));
   }
   
   // Save position state if open
   string statusfilename;
   for(int s=0; s<numStragtegies; s++){
      statusfilename = StringFormat("%sPOSSTATE%d.txt",directoryNameComm,s);
      if (positions[s]!=0){
         int fh = FileOpen(statusfilename,FILE_WRITE|FILE_CSV|FILE_ANSI,',');
         FileWrite(fh,positions[s],deadlines[s],Bis[s],Ais[s],difs_ticks[s],nEventsPerStats[s],stoplosses[s],takeprofits[s],pos_tickets[s],lot);
         FileClose(fh);
      }
   }
   
  }
  


//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
  
   /*****************************
   *********** FETCHER **********
   ******************************/
   
   // new Event comes. Get bids, asks and datetimes
   
   currTime = TimeCurrent();
   bid = SymbolInfoDouble(Symbol(),SYMBOL_BID);
   ask = SymbolInfoDouble(Symbol(),SYMBOL_ASK);
   
   ticks_counter = ticks_counter+1;
   /*
   if(position==0 && first_pos==0){
      Print("Bid ",bid," ask ",ask," Ticks counter ",ticks_counter);
   }*/

   //ArrayPrint(bid_counters);
   
   // init and open new buffer
   if(bids_counter==0){
      // Generate file name
      filename = thisSymbol+"_"+StringFormat("%02d",file_index)+".txt";
         
      //Print("Open "+filename);
      // init file
      filehandle = FileOpen(directoryNameLive+filename,FILE_WRITE|FILE_CSV|FILE_ANSI,',');
      FileWrite(filehandle,"DateTime","SymbolBid","SymbolAsk");
      saveAccountInfo();
      if(n_pos_open>0){
         savePositionState();
      }
      //if(position==0 && first_pos==0){
       //  Print("Ticks counter ",ticks_counter);
      //}
   }
   
   // save new entry in buffer
   FileWrite(filehandle,TimeCurrent(),bid,ask);
   
   buffer[bids_counter][0] = bid;
   buffer[bids_counter][1] = ask;
   
   // update bids counter
   bids_counter = (bids_counter+1)%bufferSize;
   
   // close buffer if bids counter reset
   if(bids_counter==0){
      //Print("Close "+filename);
      tic = TimeCurrent();
      ticks_counter_open = ticks_counter;
      FileClose(filehandle);  
      file_index = (file_index+1)%nBuffers;
   }
   
   /****************************
   ********** RECORDER *********
   *****************************/
   
   // permanent-record data
   if(record_data){
      
      recordData();
      
   }
   
   /****************************
   *********** TRADER **********
   *****************************/
   
   
   //checkForOpening();
   
   controlPositionFlow();
   
  }
  
void OnTimer(){
   
   // check if flag close position has been launched
   string lcfilename;
   for(int s=0; s<numStragtegies; s++){
      lcfilename = StringFormat("%s%s%d",directoryNameLive,LCfile,s);
      if(FileIsExist(lcfilename)==1 ){
         int k;
         do{
            filehandle_trader = FileOpen(lcfilename,FILE_READ|FILE_ANSI);
            predictionString = FileReadString(filehandle_trader);
            k = StringSplit(predictionString,StringGetCharacter(",",0),chunks);
            FileClose(filehandle_trader);
            //tries = tries+1;
         }while(k!=1);
         // TEMP: define strategy index as 0
         int str_idx = (int)StringToInteger(chunks[0]);
         string message = "LC found";
         Print(message);
         //writeLog(message);
         while(!FileDelete(lcfilename));
         if(positions[str_idx]!=0){
            close_type = "LC";
            closePosition(str_idx);
         }
         else{
            message = "WARNING! Position not opened. LC skipped";
            Print(message);
            //writeLog(message);
         }
       }
      checkForOpening(s);
    }
    

}

void savePositionState(){
   //--- we will look for the position by the symbol of the chart, on which the EA is working
   //string symbol=Symbol();
//--- attempt to get the position
   for(int s=0; s<numStragtegies; s++){
      if (pos_tickets[s]!=-1){
         bool selected = PositionSelectByTicket(pos_tickets[s]);
         if(selected) // if the position is selected
           {
            long pos_id            =PositionGetInteger(POSITION_IDENTIFIER);
            double price           =PositionGetDouble(POSITION_PRICE_OPEN);
            double current_profit  =PositionGetDouble(POSITION_PROFIT);
            double volume          =PositionGetDouble(POSITION_VOLUME);
            double current_price   =PositionGetDouble(POSITION_PRICE_CURRENT);
            double swap            =PositionGetDouble(POSITION_SWAP);
            long pos_magic         =PositionGetInteger(POSITION_MAGIC);
            
            //PrintFormat("Position #%d by %s: POSITION_MAGIC=%d, price=%G, current_profit=%G",
            //           pos_id, thisSymbol, pos_magic, price, current_profit);
            // save info
            string infofilename = StringFormat("%sPOSINFO%d.txt",directoryNameComm,s);
            int fh = FileOpen(infofilename,FILE_WRITE|FILE_CSV|FILE_ANSI,',');
            if(fh>0){
               // TODO! ADD DIRECTION
               FileWrite(fh,pos_id,volume,price,current_price,current_profit,swap,nEventsPerStats[s]-difs_ticks[s]-deadlines[s]-1,positions[s]);
               FileClose(fh);
            }
         }
         else        // if selecting the position was unsuccessful
           {
            PrintFormat("Unsuccessful selection of the position by the symbol %s. Error",thisSymbol,GetLastError());
            int postionsTotal=PositionsTotal();
            if(sot>-1){
               
               ulong ticket;
               for(int p=postionsTotal-1;p>=0;p--){
                  if(sot>-1){
                     ticket = PositionGetTicket(p);
                     selected = PositionSelectByTicket(ticket);
                     bool ticketTaken = false;
                     if(selected && PositionGetString(POSITION_SYMBOL)==thisSymbol){
                        for(int t=0;t<numStragtegies;t++){
                           if(pos_tickets[t]==ticket)ticketTaken = true;
                        }
                        if(!ticketTaken){
                           pos_tickets[sot] = ticket;
                           PrintFormat("Ticket %d found and added to strategy ",ticket);
                           sot = -1;
                        }
                        else Print("WARNING! Ticket taken. Skipped assigment.");
                     }
                  }
               }
               if(sot!=-1)Print("WARNING! sot!=-1 but no ticket found");
            }
         }
      }
   }
}

void saveAccountInfo()
  {
//--- trade server name
   string server=AccountInfoString(ACCOUNT_SERVER);
//--- account number
   int login=(int)AccountInfoInteger(ACCOUNT_LOGIN);
//--- long value output
   long leverage=AccountInfoInteger(ACCOUNT_LEVERAGE);
   //PrintFormat("%s %d: leverage = 1:%I64d",
   //            server,login,leverage);
//--- account currency
   string currency=AccountInfoString(ACCOUNT_CURRENCY);
//--- double value output with 2 digits after the decimal point
   double equity=AccountInfoDouble(ACCOUNT_EQUITY);
   //PrintFormat("%s %d: account equity = %.2f %s",
    //           server,login,equity,currency);
   double balance=AccountInfoDouble(ACCOUNT_BALANCE);
   //PrintFormat("%s %d: account balance = %.2f %s",
   //            server,login,balance,currency);               
//--- double value output with mandatory output of the +/- sign
   double profits=AccountInfoDouble(ACCOUNT_PROFIT);
   //PrintFormat("%s %d: current result for open positions = %+.2f %s",
   //            server,login,profits,currency);
//--- double value output with variable number of digits after the decimal point
   //double point_value=SymbolInfoDouble(_Symbol,SYMBOL_POINT);
   //string format_string=StringFormat("%%s: point value  = %%.%df",_Digits);
   //PrintFormat(format_string,_Symbol,point_value);
//--- int value output
   //int spreads=(int)SymbolInfoInteger(_Symbol,SYMBOL_SPREAD);
   //string message = StringFormat("Equity %.2f Balance %.2f Profits %.2f",equity,balance,profits);
   //Print(message);
   //writeLog(message);
   int fh = FileOpen(directoryNameAccount+"Status.txt",FILE_WRITE|FILE_CSV|FILE_ANSI,',');
   if(fh>0){
      FileWrite(fh,balance,leverage,equity,profits);
      FileClose(fh);
   }//else Print("WARNING! Error when opeing Status.txt file. Status not updated");
   //PrintFormat("%s: current spread in points = %d ",
   //            _Symbol,spreads);
//--- double value output in the scientific (floating point) format with 17 meaningful digits after the decimal point
   //PrintFormat("DBL_MAX = %.17e",DBL_MAX);
//--- double value output in the scientific (floating point) format with 17 meaningful digits after the decimal point
   //PrintFormat("EMPTY_VALUE = %.17e",EMPTY_VALUE);
//--- output using PrintFormat() with default accuracy
   //PrintFormat("PrintFormat(EMPTY_VALUE) = %e",EMPTY_VALUE);
//--- simple output using Print()
   //Print("Print(EMPTY_VALUE) = ",EMPTY_VALUE);
  }
  
//+------------------------------------------------------------------+
//| TradeTransaction function                                        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
//--- get transaction type as enumeration value 
   ENUM_TRADE_TRANSACTION_TYPE type=trans.type;
   
//--- if transaction is result of addition of the transaction in history
   if(type==TRADE_TRANSACTION_DEAL_ADD)
   {
      long     deal_entry        =0;
      double   deal_profit       =0.0;
      double   deal_volume       =0.0;
      double   swap              =0.0;
      string   deal_symbol       ="";
      long     deal_magic        =0;
      long     deal_reason       =-1;
      
      if(HistoryDealSelect(trans.deal))
        {
         deal_entry =HistoryDealGetInteger(trans.deal,DEAL_ENTRY);
         deal_profit=HistoryDealGetDouble(trans.deal,DEAL_PROFIT);
         deal_volume=HistoryDealGetDouble(trans.deal,DEAL_VOLUME);
         deal_symbol=HistoryDealGetString(trans.deal,DEAL_SYMBOL);
         deal_magic =HistoryDealGetInteger(trans.deal,DEAL_MAGIC);
         deal_reason=HistoryDealGetInteger(trans.deal,DEAL_REASON);
         swap       =HistoryDealGetDouble(trans.deal,DEAL_SWAP);
        }
      else
         return;
      
      //if(deal_symbol==m_symbol.Name() && deal_magic==EXPERT_MAGIC)
      if(deal_entry==DEAL_ENTRY_OUT && deal_symbol==thisSymbol)
      {
         saveAccountInfo();
         real_profit = deal_profit;
         // calulate real GROI and ROI
         //profit = ROI*lot*lot_in_eur/100;
         
         
         
         double equity = AccountInfoDouble(ACCOUNT_EQUITY);
         // check if close type is manual
         if (close_type==""){
            close_type = "MA";
            for(int s=0; s<numStragtegies; s++){
               if(positions[s]!=0)sit=s;
            }
            
            
         }else{
            if (positions[sit]==1){
               GROI = 100*(ask-Ais[sit])/Ais[sit];
               ROI = 100*(bid-Ais[sit])/Ais[sit];
         
            }else{if(positions[sit]==-1){
               GROI = 100*(Bis[sit]-bid)/ask;
               ROI = 100*(Bis[sit]-ask)/ask;
            }}
            
            profit = ROI*lot*lot_in_eur/100;
            spread = GROI-ROI;
         }
         if(sit!=-1){
            ROI = 100*real_profit/(deal_volume*lot_in_eur);
            GROI = ROI+spread;
            //PrintFormat("Real profit: %.2f",real_profit);
            string message = StringFormat("%s %d Bi %.4f BiS %.4f Ai %.4f AiS %.4f Bo %.4f Ao %.4f SP %.4f GROI %.4f ROI %.4f Profit %.2f ticks %d real profit %.2f budget %.2f swap %.2f str_idx %d",
                                          close_type,positions[sit],Bis[sit],Bi_solls[sit],Ais[sit],Ai_solls[sit],bid,ask,spread,GROI,ROI,profit,difs_ticks[sit],real_profit,equity,swap,sit);
            Print(message);
            //writeLog(message);
            // send signal to trading manager that position has been closed
            if(close_type=="LC"){
                filename = "CL";
                     
            }else{
                filename = close_type;
            }
            filehandleTest = FileOpen(directoryNameLive+filename,FILE_WRITE|FILE_CSV|FILE_ANSI,',');
            FileWrite(filehandleTest,thisSymbol,toc,TimeCurrent(),positions[sit],Bis[sit],Ais[sit],bid,ask,difs_ticks[sit],GROI,spread,ROI,real_profit,equity,swap,sit);
            FileClose(filehandleTest);
            
            string infofilename = StringFormat("%sPOSINFO%d.txt",directoryNameComm,sit);
            if(FileIsExist(infofilename)==1 ){
               while(!FileDelete(infofilename));
             }
             
            if(FileIsExist(directoryNameComm+"WARNING")==1 ){
               while(!FileDelete(directoryNameComm+"WARNING"));
            }
            deadlines[sit] = -1;
            positions[sit] = 0;
            pos_tickets[sit] = -1;
            sit = -1;
            close_type = "";
            n_pos_open = n_pos_open-1;
            closingInProgress = false;
         }
         /*
         if(deal_reason==DEAL_REASON_SL){
            //ExtLot*=2.0;
            Print("DEAL_REASON_SL");
            writeLog("DEAL_REASON_SL");}
         else if(deal_reason==DEAL_REASON_TP){
            //ExtLot=m_symbol.LotsMin();
            Print("DEAL_REASON_TP");
            writeLog("DEAL_REASON_TP");}*/
      }
      
   }else{
      if(type==TRADE_TRANSACTION_ORDER_ADD && thisSymbol==trans.symbol){
         ulong                         deal=trans.deal;             // Deal ticket
         ulong                         order=trans.order;            // Order ticket
         string                        symbol=trans.symbol;
         ulong                         position=trans.position;
         int                           postionsTotal = PositionsTotal();
         Print(StringFormat("deal %d order %d symbol %s ticket %d total_pos %d",deal, order, symbol, position, postionsTotal));
         if(sot>-1){
            bool selected;
            ulong ticket;
            for(int p=postionsTotal-1;p>=0;p--){
               if(sot>-1){
                  ticket = PositionGetTicket(p);
                  selected = PositionSelectByTicket(ticket);
                  bool ticketTaken = false;
                  if(selected && PositionGetString(POSITION_SYMBOL)==thisSymbol){
                     for(int t=0;t<numStragtegies;t++){
                        if(pos_tickets[t]==ticket)ticketTaken = true;
                     }
                     if(!ticketTaken){
                        pos_tickets[sot] = ticket;
                        PrintFormat("Ticket %d found and added to strategy ",ticket);
                        sot = -1;
                     }
                     else Print("WARNING! Ticket taken. Skipped assigment.");
                  }
               }
            }
            if(sot!=-1)Print("WARNING! sot!=-1 but no ticket found");
         }
      }
    }
 }
//+------------------------------------------------------------------+
//|                                           fetcher_trader_v10.mq5 |
//|                                                       mgutierrez |
//|                                       https://www.kaissandra.com |
//+------------------------------------------------------------------+
#property copyright "mgutierrez"
#property link      "https://www.kaissandra.com"
#property version   "1.00"
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
double Bi;
double Ai=1;
double Bi_soll;
double Ai_soll;
double Bo_soll;
double Ao_soll;
double real_profit;

string filename;
string logfilename;
string thisSymbol;
string directoryNameLive;
string directoryNameLog;
string directoryNameRecordings;
string directoryNameAccount;
string chunks[];

// Define variables
CTrade m_Trade;
CPositionInfo m_Position;
//input double Lot = 0.01;
double lot;
const double lot_in_eur = 100000;
bool record_data = true;
bool closingInProgress;

long deadline = -1;
int filehandle_trader;
int filehandlelog;
long nEventsPerStat;
string predictionString;
long prediction;
string TTfile;
string LCfile;
bool timer_bool;
string close_type;
//datetime currTime;

double stoploss;
double takeprofit;
double sl_protect;
double tp_protect;
double sl_protect_prev;
double tp_protect_prev;
double B_sl;
double A_sl;
double sl_thr = 1;//0.0001; //in ratio (1 pip=0.0001)
double slThrPips = 100;
double tpThrPips = 100;
double slProThrPips = 100;
double const PIP = 0.0001;
//double bid;
//double ask;
double inter_profit;
double profit;
double GROI;
double ROI;
double spread;
int position = 0; // type of position opened: 1 long/-1 short
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
long dif_ticks = 0;

//int firstBuff[2] = {1,1};

// Strategy variables


void openPosition(string origin, int thisPos){
   
   position = thisPos;
   Bi = bid;
   Ai = ask;
   B_sl = Bi;
   A_sl = Ai;
   string thisPosString;
   string message;
   if(thisPos==1){
      while(!m_Trade.Buy(lot,thisSymbol)){
         //string message = StringFormat();
         message = StringFormat("WARNING! Buy -> false. Result Retcode: %u, description of result: %s",m_Trade.ResultRetcode(),m_Trade.ResultRetcodeDescription());
         Print(message);
         //writeLog(message);
         //Print("WARNING! Buy -> false. Result Retcode: ",m_Trade.ResultRetcode(),", description of result: ",m_Trade.ResultRetcodeDescription());
      }
      //string message = StringFormat("Buy -> talse. Result Retcode: ",m_Trade.ResultRetcode(),", description of result: ",m_Trade.ResultRetcodeDescription());
      message = StringFormat("Buy -> true. Result Retcode: %u, description of result: %s",m_Trade.ResultRetcode(),m_Trade.ResultRetcodeDescription());
      //Print("Buy -> true. Result Retcode: ",m_Trade.ResultRetcode(),", description of result: ",m_Trade.ResultRetcodeDescription());
      Print(message);
      //writeLog(message);
      //stoploss = -Ai*sl_thr+Bi;
      stoploss = updateSL(Ai, thisPos, slThrPips);
      takeprofit = updateTP(Ai, thisPos, tpThrPips);
      thisPosString = "Long";
   }
   else{if(thisPos==-1){
      while(!m_Trade.Sell(lot,thisSymbol)){
         //string message = StringFormat();
         message = StringFormat("WARNING! Sell -> false. Result Retcode: %u, description of result: %s",m_Trade.ResultRetcode(),m_Trade.ResultRetcodeDescription());
         Print(message);
        // writeLog(message);
         //Print("WARNING! Sell -> false. Result Retcode: ",m_Trade.ResultRetcode(),", description of result: ",m_Trade.ResultRetcodeDescription());
      }
      message = StringFormat("Sell -> true. Result Retcode: %u, description of result: %s",m_Trade.ResultRetcode(),m_Trade.ResultRetcodeDescription());
      Print(message);
      //writeLog(message);
      //Print("Sell -> true. Result Retcode: ",m_Trade.ResultRetcode(),", description of result: ",m_Trade.ResultRetcodeDescription());
      //stoploss = Bi*sl_thr+Ai;
      stoploss = updateSL(Bi, thisPos, slThrPips);
      takeprofit = updateTP(Bi, thisPos, tpThrPips);
      thisPosString = "Short";
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
   deadline = 1; // reset deadline
   Bi_soll = buffer[bufferSize-1][0];
   Ai_soll = buffer[bufferSize-1][1];
   dif_ticks = ticks_counter-ticks_counter_open;
   message = StringFormat("%d. #Events %d Ticks %d BiS %.4f BiI %.4f AiS %.4f AiI %.4f SL %.4f SP %.4f",thisPos,nEventsPerStat,dif_ticks,Bi_soll, 
         Bi,Ai_soll,Ai,stoploss,(Ai-Bi)/Ai);
   Print(message);
   //writeLog(message);
   //Print(origin," ",thisPosString,". Number events ",nEventsPerStat," Ticks diff ",dif_ticks," Bi soll ",Bi_soll, " Bi ist ",Bi," Ai soll ",Ai_soll," Ai ",Ai," SL ",stoploss, " SP ",(Ai-Bi)/Ai);
}


void closePosition(){
   int executed = 0;
   //bool closed = false;
   string message;
   while(!executed && position!=0){
      if(m_Trade.PositionClose(thisSymbol))
         executed = 1;
      message = StringFormat("Close position executed: %d",executed);
      Print(message);
      //writeLog(message);
      //if(closed && m_Trade.ResultDeal()!=0)
      //   executed = true;
   }// close position if deadline reached
   if(position!=0){
      message = StringFormat("Close -> true. Result Retcode: %u, description of result: %s",m_Trade.ResultRetcode(),m_Trade.ResultRetcodeDescription());
      Print(message);
      //writeLog(message);
      deadline = -1;
      closingInProgress = true;
      
      if (position==1){
         GROI = 100*(ask-Ai)/Ai;
         ROI = 100*(bid-Ai)/Ai;
   
      }else{if(position==-1){
         GROI = 100*(Bi-bid)/ask;
         ROI = 100*(Bi-ask)/ask;
      }}
      
      profit = ROI*lot*lot_in_eur/100;
      spread = GROI-ROI;
   }
   else{
      message = "WARNING! Try to close position but no position is open. Skipped";
      Print(message);
      //writeLog(message);
   }

}
//+------------------------------------------------------------------+
//| Check for opening                                                |
//+------------------------------------------------------------------+
void checkForOpening(){
//---
   // Launch position 
   string message;
      if(FileIsExist(directoryNameLive+TTfile)==1 && !closingInProgress){
         message = "TT found";
         Print(message);
         //writeLog(message);
         // open position
            first_pos = 1;
            
            
            // Read type of trigger and delete TT
            int k;
            bool reset = false;
            int tries = 0;
            do{
               filehandle_trader = FileOpen(directoryNameLive+TTfile,FILE_READ|FILE_ANSI);
               predictionString = FileReadString(filehandle_trader);
               k = StringSplit(predictionString,StringGetCharacter(",",0),chunks);
               FileClose(filehandle_trader);
               tries = tries+1;
            }while(k!=4);
            prediction = StringToInteger(chunks[0]);
            lot = StringToDouble(chunks[1]);
            slThrPips = (int)StringToInteger(chunks[3]);
            //Print("slThrPips %d",slThrPips);
            // extend deadline only when new deadline is further in time
            if(nEventsPerStat-deadline<StringToDouble(chunks[2])){
               nEventsPerStat = StringToInteger(chunks[2]);
               reset = true;
            }
            
            FileDelete(directoryNameLive+TTfile);
            long thr_pred = 0;
            //diffTime = toc-tic;
            // launch long positon
            //Print(prediction);
            if(prediction>thr_pred){
               //if the position for this symbol already exists -> extend deadline
               if(m_Position.Select(thisSymbol)){
                  
                  //if(m_Position.PositionType()==POSITION_TYPE_SELL) m_Trade.PositionClose(my_symbol);  //and this is a Sell position, then close it
                  if(m_Position.PositionType()==POSITION_TYPE_BUY){
                     if(reset){
                        message = StringFormat("Deadline extended by %d",nEventsPerStat);
                        Print(message);
                        //writeLog(message);
                        //dif_ticks = dif_ticks+ticks_counter-ticks_counter_open;
                        // reset deadline
                        deadline = 1;
                     }else{
                        message = StringFormat("Deadline not extended. Remaining %d",nEventsPerStat-dif_ticks-deadline);
                        Print(message);
                        //writeLog(message);
                     }
                     
                  }
               }else{ // open new long position
                  toc = TimeCurrent();
                  openPosition("TICK", 1);
                  }
               
            }
            // launch short positon
            else{if(prediction<thr_pred){
               //if the position for this symbol already exists -> extend deadline
               if(m_Position.Select(thisSymbol)){
                  
                  if(m_Position.PositionType()==POSITION_TYPE_SELL){ 
                     if(reset){
                        //dif_ticks = dif_ticks+ticks_counter-ticks_counter_open;
                        message = StringFormat("Deadline extended by %d",nEventsPerStat);
                        Print(message);
                        //writeLog(message);
                        // reset deadline
                        deadline = 1;
                     }else{
                        message = StringFormat("Deadline not extended. Remaining %d",nEventsPerStat-dif_ticks-deadline);
                        Print(message);
                        //writeLog(message);
                     }
                  }
               }
               // open new position
               else{
                  toc = TimeCurrent();
                  openPosition("TICK", -1);
               }
               
               // if else
               }else{
                        message = StringFormat("ERROR!! Prediction cannot be zero!. Prediction string %s. Tries %d. Prediction %d",predictionString,tries,prediction);
                        Print(message);
                        //writeLog(message);
                     }
            }
            
      }
      else{if(FileIsExist(directoryNameLive+TTfile)==1){
         message = "WARNING! TT found but closing in progress. Opening delayed";
         Print(message);
         //writeLog(message);
      }
      
      }
   
}

void controlPositionFlow(){
   // Check state of open position
   string message;
      if(deadline>0){
         if (position==1){
            GROI = 100*(ask-Ai)/Ai;
            ROI = 100*(bid-Ai)/Ai;
            sl_protect = updateSL(ask, position, slProThrPips);
            tp_protect = updateTP(ask, position, slProThrPips);
         }else{if(position==-1){
            GROI = 100*(Bi-bid)/ask;
            ROI = 100*(Bi-ask)/ask;
            sl_protect = updateSL(bid, position, slProThrPips);
            tp_protect = updateTP(bid, position, slProThrPips);
         }}
         //Print("",nEventsPerStat-dif_ticks-deadline,"",GROI," ",ROI,"%");
         message = StringFormat("Deadline in  %d GROI %.4f ROI %.4f",nEventsPerStat-dif_ticks-deadline,GROI,ROI);
         //Print(message);
         //writeLog(message);
         if(sl_protect!=sl_protect_prev || tp_protect!=tp_protect_prev)
            m_Trade.PositionModify(thisSymbol,sl_protect,tp_protect);
         sl_protect_prev = sl_protect;
         tp_protect_prev = tp_protect;
         // update deadline
         deadline = (deadline+1)%(nEventsPerStat-dif_ticks);
         
         // close position if deadline reached
         if (deadline==0){
            message = "Deadline reached";
            Print(message);
            //writeLog(message);
            close_type = "CL";
            closePosition();
         }
         
         else{
            // check if stoploss reached
            if((position==1 && ask<stoploss) || (position==-1 && bid>stoploss)){
               message = "SL reached";
               Print(message);
               //writeLog(message);
               close_type = "SL";
               closePosition();
            }
            // check if takeprofit reached
            if((position==1 && ask>takeprofit) || (position==-1 && bid<takeprofit)){
               message = "TP reached";
               Print(message);
               //writeLog(message);
               // WARNING! Temporal close_type as SL. TP not implemented yet!!!
               close_type = "SL";
               closePosition();
            }
         }
         
         //accountInfo();
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
   file_index = 0;
   thisSymbol = Symbol();
   directoryNameLive = "IOlive//"+thisSymbol+"//";
   directoryNameLog = "Log//"+thisSymbol+"//";
   directoryNameAccount = "Account//";
   directoryNameRecordings = "Data//"+thisSymbol+"//";
   //Print(thisSymbol+" fetcher launched");
   // init bid and ask to avoud division by zero
   bid = 1.0;
   ask = 1.0;
   ulong m_slippage = 100;                // slippage
   
   timer_bool = EventSetMillisecondTimer(10);
   m_Trade.SetDeviationInPoints(m_slippage);

   TTfile = "TT";// trade trigger file signalling
   LCfile = "LC";
   // set milisecond timer
   //timer_bool = EventSetMillisecondTimer(10);
   Print(thisSymbol+" Fetcher-Trader-Recorder version 2.0 launched");
   
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
   //Print(stringThisDate);
   // init file
   filehandle_record=FileOpen(directoryNameRecordings+"//"+filename,FILE_WRITE|FILE_CSV|FILE_ANSI,',');
   FileWrite(filehandle_record,"DateTime","SymbolBid","SymbolAsk");
   // init log file
   logfilename = thisSymbol+"_"+stringThisDate+".log";
   filehandlelog = FileOpen(directoryNameLog+logfilename,FILE_WRITE|FILE_CSV|FILE_ANSI);
   
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
   // check if position is opened
   if (m_Position.Select(thisSymbol)){
      m_Trade.PositionClose(thisSymbol);// close position
   }
   FileClose(filehandlelog);
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
   
   
   checkForOpening();
   
   controlPositionFlow();
   
  }
  
void OnTimer(){
   
   // check if flag close position has been launched
   if(FileIsExist(directoryNameLive+LCfile)==1 ){
      string message = "LC found";
      Print(message);
      //writeLog(message);
      while(!FileDelete(directoryNameLive+LCfile));
      if(position!=0){
         close_type = "LC";
         closePosition();
      }
      else{
         message = "WARNING! Position not opened. LC skipped";
         Print(message);
         //writeLog(message);
      }
    }
    checkForOpening();

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
   string message = StringFormat("Equity %.2f Balance %.2f Profits %.2f",equity,balance,profits);
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
      string   deal_symbol       ="";
      long     deal_magic        =0;
      long     deal_reason       =-1;
      
      if(HistoryDealSelect(trans.deal))
        {
         deal_entry=HistoryDealGetInteger(trans.deal,DEAL_ENTRY);
         deal_profit=HistoryDealGetDouble(trans.deal,DEAL_PROFIT);
         deal_volume=HistoryDealGetDouble(trans.deal,DEAL_VOLUME);
         deal_symbol=HistoryDealGetString(trans.deal,DEAL_SYMBOL);
         deal_magic=HistoryDealGetInteger(trans.deal,DEAL_MAGIC);
         deal_reason=HistoryDealGetInteger(trans.deal,DEAL_REASON);
         
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
         ROI = 100*real_profit/(lot*lot_in_eur);
         GROI = ROI+spread;
         
         double equity = AccountInfoDouble(ACCOUNT_EQUITY);
         //PrintFormat("Real profit: %.2f",real_profit);
         string message = StringFormat("%s %d Bi %.4f BiS %.4f Ai %.4f AiS %.4f Bo %.4f Ao %.4f SP %.4f GROI %.4f ROI %.4f Profit %.2f ticks %d real profit %.2f budget %.2f",
                                       close_type,position,Bi,Bi_soll,Ai,Ai_soll,bid,ask,spread,GROI,ROI,profit,dif_ticks,real_profit,equity);
         Print(message);
         //writeLog(message);
         // send signal to trading manager that position has been closed
         if(close_type=="LC"){
             filename = "CL";
                  
         }else{
             filename = close_type;
         }
         filehandleTest = FileOpen(directoryNameLive+filename,FILE_WRITE|FILE_CSV|FILE_ANSI,',');
         FileWrite(filehandleTest,thisSymbol,toc,TimeCurrent(),position,Bi,Ai,bid,ask,dif_ticks,GROI,spread,ROI,real_profit,equity);
         FileClose(filehandleTest);
         position = 0;
         closingInProgress = false;
         
         if(deal_reason==DEAL_REASON_SL){
            //ExtLot*=2.0;
            Print("DEAL_REASON_SL");
            writeLog("DEAL_REASON_SL");}
         else if(deal_reason==DEAL_REASON_TP){
            //ExtLot=m_symbol.LotsMin();
            Print("DEAL_REASON_TP");
            writeLog("DEAL_REASON_TP");}
      }
      
   }
 }
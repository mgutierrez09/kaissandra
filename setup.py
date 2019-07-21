import sys
import os
#from setuptools import setup, find_packages

if __name__=='__main__':
    path = os.getcwd()+'\\'
    if path not in sys.path:
        sys.path.insert(0, path)
        #sys.path.append(path)
        print(path+" added to python path")
    else:
        print(path+" already added to python path")
    
    if 'build' in sys.argv:
        import datetime as dt
        py_directories = ['IO/','config/','weights/','results/','resultsLive/','IOlive/','DB/','log/','Data/','Data_test/','DB/stats/','gre/']
        
        if os.environ.get('MT5_ROOT_DIR')!=None:
            mt5_root_dir = os.environ.get('MT5_ROOT_DIR')
        else:
            raise ValueError("MT5_ROOT_DIR environment variable has not been set. Set it before running config")
        print(mt5_root_dir)
        if not os.path.exists(mt5_root_dir):
            try:
                os.mkdir(mt5_root_dir)
                print(mt5_root_dir+" created")
                mt5_root_dir_exists = True
            except:
                print("WARNING! "+mt5_root_dir+" could not be created. Revise MT5_ROOT_DIR environment variable.")
                mt5_root_dir_exists = False
        else:
            mt5_root_dir_exists = True
        mt5_directories = ['Account','IOlive','Data','Log']
        root_path = '/'.join(path.split('\\')[:-2])+'/'
        for directory in py_directories:
            dirpath = root_path+directory
            # create default directory tree
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)
                print(dirpath+" created")
            else:
                print(dirpath+" already exists")
        # create MT5 directories
        for directory in mt5_directories:
            dirpath = mt5_root_dir+directory
            # create default directory tree
            if not os.path.exists(dirpath) and mt5_root_dir_exists:
                try:
                    os.mkdir(dirpath)
                    print(dirpath+" created")
                except:
                    print("WARNING! "+dirpath+" could not be created. Revise MT5_ROOT_DIR environment variable.")
            elif os.path.exists(dirpath):
                print(dirpath+" already exists")
    # build local_config.py file
        config_file = ("#Automatically created on "+str(dt.datetime.now())+"\n\n"+
            "\nclass Local:\n"+
            "\tdirectory_MT5 = '"+mt5_root_dir+"'\n"+
            "\tdirectory_MT5_IO = directory_MT5+'IOlive/'\n"+
            "\tdirectory_MT5_log = directory_MT5+'Log/'\n"+
            "\tdirectory_MT5_account = directory_MT5+'Account/'\n"+
            "\thdf5_directory = '"+root_path+py_directories[6]+"'\n"+
            "\troot_dir = '"+'/'.join(path.split('\\')[:-3])+'/'+"'\n"+
            "\tstats_directory = '"+root_path+py_directories[10]+"'\n"+
            "\tIO_directory = '"+root_path+py_directories[0]+"'\n"+
            "\tweights_directory = '"+root_path+py_directories[2]+"'\n"+
            "\tresults_directory = '"+root_path+py_directories[3]+"'\n"+
            "\tlive_results_dict = '"+root_path+py_directories[4]+"'\n"+
            "\tio_live_dir = '"+root_path+py_directories[5]+"'\n"+
            "\tconfig_directory = '"+root_path+py_directories[1]+"'\n"+
            "\tlog_directory = '"+root_path+py_directories[7]+"'\n"+
            "\tdata_dir = '"+root_path+py_directories[8]+"'\n"+
            "\tdata_test_dir = '"+root_path+py_directories[9]+"'\n"+
            "\tgre_directory = '"+root_path+py_directories[11]+"'\n"+
            "local_vars = Local()\n")
        local_config_filepath = path+'kaissandra\\local_config.py'
        if not os.path.exists(local_config_filepath):
            f = open(local_config_filepath,'w+')
            f.write(config_file)
            f.close()
            print(local_config_filepath+" created")
        else:
            print(local_config_filepath+" already exists")
#setup(
#    name="kaissandra",
#    version="0.0",
#    packages=find_packages()
#)
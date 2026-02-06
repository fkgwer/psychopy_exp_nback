#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Fri Feb  6 15:25:22 2026
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '4'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from eeg
import pyxid2
import threading
import signal


def exit_after(s):
    '''
    function decorator to raise KeyboardInterrupt exception
    if function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, signal.raise_signal, args=[signal.SIGINT])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


@exit_after(1)  # exit if function takes longer than 1 seconds
def _get_xid_devices():
    return pyxid2.get_xid_devices()


def get_xid_devices():
    print("Getting a list of all attached XID devices...")
    attempt_count = 0
    while attempt_count >= 0:
        attempt_count += 1
        print('     Attempt:', attempt_count)
        attempt_count *= -1  # try to exit the while loop
        try:
            devices = _get_xid_devices()
        except KeyboardInterrupt:
            attempt_count *= -1  # get back in the while loop
    return devices


devices = get_xid_devices()

if devices:
    dev = devices[0]
    print("Found device:", dev)
    assert dev.device_name in ['Cedrus C-POD', 'Cedrus StimTracker Quad'], "Incorrect XID device detected."
    if dev.device_name == 'Cedrus C-POD':
        pod_name = 'C-POD'
    else:
        pod_name = 'M-POD'
    dev.set_pulse_duration(50)  # set pulse duration to 50ms

    # Start EEG recording
    print("Sending trigger code 126 to start EEG recording...")
    dev.activate_line(bitmask=126)  # trigger 126 will start EEG
    print("Waiting 10 seconds for the EEG recording to start...\n")
    core.wait(10)  # wait 10s for the EEG system to start recording

    # Marching lights test
    print(f"{pod_name}<->eego 7-bit trigger lines test...")
    for line in range(1, 8):  # raise lines 1-7 one at a time
        print("  raising line {} (bitmask {})".format(line, 2 ** (line-1)))
        dev.activate_line(lines=line)
        core.wait(0.5)  # wait 500ms between two consecutive triggers
    dev.con.set_digio_lines_to_mask(0)  # XidDevice.clear_all_lines()
    print("EEG system is now ready for the experiment to start.\n")

else:
    # Dummy XidDevice for code components to run without C-POD connected
    class dummyXidDevice(object):
        def __init__(self):
            pass
        def activate_line(self, lines=None, bitmask=None):
            pass


    print("WARNING: No C/M-POD connected for this session! "
          "You must start/stop EEG recording manually!\n")
    dev = dummyXidDevice()

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'auditory_n_back'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [2992, 1934]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/wernstal/Library/Mobile Documents/com~apple~CloudDocs/Forskning/My_PhD:Patrick Purdon/Code/Psychopy/1.n-back_w_practice_latest_post_Alex_comments/auditory_n_back.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # create speaker 'n_back_instructions'
    deviceManager.addDevice(
        deviceName='n_back_instructions',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'sound_to_match_if_0_back'
    deviceManager.addDevice(
        deviceName='sound_to_match_if_0_back',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'ir_zero_otherwise_press'
    deviceManager.addDevice(
        deviceName='ir_zero_otherwise_press',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('end_routine') is None:
        # initialise end_routine
        end_routine = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_routine',
        )
    # create speaker 'press_to_continue_prompt'
    deviceManager.addDevice(
        deviceName='press_to_continue_prompt',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('force_advance') is None:
        # initialise force_advance
        force_advance = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='force_advance',
        )
    # create speaker 'practice_indicator_sound'
    deviceManager.addDevice(
        deviceName='practice_indicator_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('force_end_of_routine_if_pressed_5') is None:
        # initialise force_end_of_routine_if_pressed_5
        force_end_of_routine_if_pressed_5 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='force_end_of_routine_if_pressed_5',
        )
    # create speaker 'sound_1'
    deviceManager.addDevice(
        deviceName='sound_1',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('force_advance_2') is None:
        # initialise force_advance_2
        force_advance_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='force_advance_2',
        )
    if deviceManager.getDevice('force_end_of_routine_if_pressed_2') is None:
        # initialise force_end_of_routine_if_pressed_2
        force_end_of_routine_if_pressed_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='force_end_of_routine_if_pressed_2',
        )
    # create speaker 'fb_sound'
    deviceManager.addDevice(
        deviceName='fb_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'thank_you'
    deviceManager.addDevice(
        deviceName='thank_you',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Settings" ---
    # Run 'Begin Experiment' code from code_exp_settings
    import numpy as np
    from scipy.stats import norm
    
    # Single RNG for this session
    rng = np.random.default_rng()
    
    # ======================================================
    # ENABLE/DISABLE PRACTICE BLOCKS
    # ======================================================
    USE_PRACTICE = True  # Set to False to skip practice entirely
    
    # ======================================================
    # PAIR PRACTICE WITH MAIN BLOCKS
    # ======================================================
    PAIR_PRACTICE_WITH_MAIN = True  # Set to True to run practice immediately before each main block
                                     # Set to False to run all practice blocks first, then all main blocks
                                     # NOTE: This only works if USE_PRACTICE = True
    
    # ======================================================
    # PHASE CONFIGURATIONS
    # ======================================================
    # Define all task phases (practice, main, etc.)
    # This is the ONLY place you need to change parameters!
    
    # Define practice configuration
    practice_config = {
        'phase_name': 'practice',
        'phase_display_name': 'Practice',
        'n_back_blocks': [0,1,2],         # Which n-back levels
        'num_trials_per_block': 5,      # Scorable trials per block
        'num_blocks_per_condition': 1,  # Repetitions of each n-back level
        'target_percentage': 1/3,       # Target trial proportion
        'zero_back_target_idx': 0,      # Index for 0-back target
        'randomization_type': 1,        # 1=shuffle then repeat, 2=full shuffle, 3=sorted, other value->no randomization
        'show_transition_screen': False # Show screen before this phase starts
    }
    
    # Define main task configuration
    main_config = {
        'phase_name': 'main',
        'phase_display_name': 'Main Task',
        'n_back_blocks': [0,1,2],
        'num_trials_per_block': 60,
        'num_blocks_per_condition': 1,
        'target_percentage': 1/3,
        'zero_back_target_idx': 0,
        'randomization_type': 2,
        'show_transition_screen': True
    }
    
    # ------------------------------------------------------
    # Inter-Trial Interval (ITI) Settings (shared across all phases)
    # ------------------------------------------------------
    iti_fixed_duration = 1.75   # To test: 1.25, 1.75, 2.25
                                # Fixed component of ITI in seconds, it already 
                                # has a minimum trial duration of stimulus_duration, set in 
                                # "code_iti_timing" in "n_back_trial" Routine
    iti_jitter_min = 0.0
    iti_jitter_max = 0.0
    
    
    
    
    # ======================================================
    # BUILD PHASE STRUCTURE BASED ON PAIRING SETTING
    # ======================================================
    
    # Safety check: Can't pair if practice is disabled
    if not USE_PRACTICE and PAIR_PRACTICE_WITH_MAIN:
        print("⚠️  WARNING: PAIR_PRACTICE_WITH_MAIN is True but USE_PRACTICE is False")
        print("   Pairing mode requires practice to be enabled. Disabling pairing.")
        PAIR_PRACTICE_WITH_MAIN = False
    
    if USE_PRACTICE and PAIR_PRACTICE_WITH_MAIN:
        # PAIRED MODE: Each practice block is paired with its corresponding main block
        print("🎯 PAIRED PRACTICE MODE ENABLED")
        print("   Each practice block will run immediately before its corresponding main block")
        
        # Find common n-back levels between practice and main
        practice_n_levels = set(practice_config['n_back_blocks'])
        main_n_levels = set(main_config['n_back_blocks'])
        
        # Determine which levels can be paired
        paired_levels = sorted(list(practice_n_levels & main_n_levels))
        practice_only_levels = sorted(list(practice_n_levels - main_n_levels))
        main_only_levels = sorted(list(main_n_levels - practice_n_levels))
        
        print(f"   Paired levels: {paired_levels}")
        if practice_only_levels:
            print(f"   Practice-only levels: {practice_only_levels}")
        if main_only_levels:
            print(f"   Main-only levels: {main_only_levels}")
        
        # Create paired phase configs
        phase_configs = []
        
        # Add practice-only blocks first (if any)
        if practice_only_levels:
            practice_only_config = practice_config.copy()
            practice_only_config['n_back_blocks'] = practice_only_levels
            practice_only_config['phase_name'] = 'practice_unpaired'
            practice_only_config['phase_type'] = 'practice'  # Add type marker
            practice_only_config['show_transition_screen'] = False
            phase_configs.append(practice_only_config)
        
        # Create paired blocks
        for n_level in paired_levels:
            # Practice block for this n-back level
            paired_practice_config = practice_config.copy()
            paired_practice_config['n_back_blocks'] = [n_level]
            paired_practice_config['phase_name'] = f'practice_{n_level}back'
            paired_practice_config['phase_display_name'] = f'Practice {n_level}-back'
            paired_practice_config['phase_type'] = 'practice'  # Add type marker
            paired_practice_config['show_transition_screen'] = False
            phase_configs.append(paired_practice_config)
            
            # Main block for this n-back level (immediately after practice)
            paired_main_config = main_config.copy()
            paired_main_config['n_back_blocks'] = [n_level]
            paired_main_config['phase_name'] = f'main_{n_level}back'
            paired_main_config['phase_display_name'] = f'Main {n_level}-back'
            paired_main_config['phase_type'] = 'main'  # Add type marker
            paired_main_config['show_transition_screen'] = True
            phase_configs.append(paired_main_config)
        
        # Add main-only blocks last (if any)
        if main_only_levels:
            main_only_config = main_config.copy()
            main_only_config['n_back_blocks'] = main_only_levels
            main_only_config['phase_name'] = 'main_unpaired'
            main_only_config['phase_type'] = 'main'  # Add type marker
            main_only_config['show_transition_screen'] = True
            phase_configs.append(main_only_config)
        
        # Now shuffle the PAIRS (not individual blocks)
        # We need to keep practice-main pairs together
        pair_indices = []
        unpaired_indices = []
        
        for i, config in enumerate(phase_configs):
            if config['phase_name'].startswith('practice_') and config['phase_name'] != 'practice_unpaired':
                # This is a paired practice block, store the pair indices
                pair_indices.append((i, i+1))  # Practice and its following main block
            elif config['phase_name'] in ['practice_unpaired', 'main_unpaired']:
                unpaired_indices.append(i)
        
        # Randomize the pairs if randomization is enabled
        if main_config['randomization_type'] in [1, 2]:
            rng.shuffle(pair_indices)
        
        # Rebuild phase_configs in the new order
        new_phase_configs = []
        
        # Add unpaired practice blocks first
        for idx in unpaired_indices:
            if phase_configs[idx]['phase_name'] == 'practice_unpaired':
                new_phase_configs.append(phase_configs[idx])
        
        # Add paired blocks in randomized order
        for practice_idx, main_idx in pair_indices:
            new_phase_configs.append(phase_configs[practice_idx])
            new_phase_configs.append(phase_configs[main_idx])
        
        # Add unpaired main blocks last
        for idx in unpaired_indices:
            if phase_configs[idx]['phase_name'] == 'main_unpaired':
                new_phase_configs.append(phase_configs[idx])
        
        phase_configs = new_phase_configs
    
    elif USE_PRACTICE:
        # TRADITIONAL MODE: All practice blocks first, then all main blocks
        practice_config['phase_type'] = 'practice'  # Add type marker
        main_config['phase_type'] = 'main'  # Add type marker
        phase_configs = [practice_config, main_config]
        print("🎯 PRACTICE MODE ENABLED (Traditional)")
        print("   All practice blocks will run first, then all main blocks")
    else:
        # NO PRACTICE MODE
        main_config['phase_type'] = 'main'  # Add type marker
        phase_configs = [main_config]
        main_config['show_transition_screen'] = False
        print("🚀 PRACTICE MODE DISABLED - Starting directly with main task")
    
    
    # ------------------------------------------------------
    # Stimulus sounds (shared across all phases)
    # ------------------------------------------------------
    stimulus_list = [
        'sine_wave_frequency128_duration0.75s_tagging_frequency43Hz_modulation_depth1.0_base_amplitude0.5.wav',
        'sine_wave_frequency256_duration0.75s_tagging_frequency43Hz_modulation_depth1.0_base_amplitude0.5.wav',
        'sine_wave_frequency512_duration0.75s_tagging_frequency43Hz_modulation_depth1.0_base_amplitude0.5.wav',
        'sine_wave_frequency880_duration0.75s_tagging_frequency43Hz_modulation_depth1.0_base_amplitude0.5.wav'
    ]
    
    # ------------------------------------------------------
    # Audio instruction files (shared)
    # ------------------------------------------------------
    instruction_files = {
        0: 'audio_instructions/next_block_info_zero_back_01.wav',
        1: 'audio_instructions/next_block_info_one_back_01.wav',
        2: 'audio_instructions/next_block_info_two_back_01.wav',
        3: 'audio_instructions/next_block_info_three_back_01.wav'
    }
    
    otherwise_press_files = {
        0: 'audio_instructions/next_block_otherwise_press_the_left_button.wav',
        1: 'audio_instructions/0_75_s_silence_48000Hz.wav',
        2: 'audio_instructions/0_75_s_silence_48000Hz.wav',
        3: 'audio_instructions/0_75_s_silence_48000Hz.wav'
    }
    
    press_to_continue_audio = 'audio_instructions/press_to_continue_audio.wav'
    thank_you_audio = 'audio_instructions/thank_you_audio.wav'
    
    # ======================================================
    # BLOCK GENERATION FUNCTION (works for any phase)
    # ======================================================
    def make_nback_block(n_back_level, n_trials, target_pct, rng, stim_list=stimulus_list, target_idx=0):
        """
        Generate a single n-back block with target and non-target trials
        
        Args:
            n_back_level: The n in n-back (0, 1, 2, 3, etc.)
            n_trials: Number of scorable trials (will add n baseline trials)
            target_pct: Proportion of trials that should be targets (e.g., 0.33)
            rng: Random number generator
            stim_list: List of stimulus filenames
            target_idx: Index of target stimulus for 0-back task
        
        Returns:
            stimulus_sequence: List of stimulus filenames
            trial_types: List of trial types ('baseline', 'target', 'non-target')
        """
        n = n_back_level
        num_trials_total = n_trials + n
        
        # Calculate target trials
        n_target_trials = int(n_trials * target_pct)
        
        # Create trial type sequence
        trial_types = ['baseline'] * n
        trial_types += ['target'] * n_target_trials
        trial_types += ['non-target'] * (num_trials_total - len(trial_types))
        
        # Shuffle only trials after first n
        remaining_trials = trial_types[n:]
        rng.shuffle(remaining_trials)
        trial_types = trial_types[:n] + remaining_trials
        
        # Generate all stimuli for this block
        stimulus_sequence = []
        
        for i, t_type in enumerate(trial_types):
            if t_type == 'target' and i >= n:
                if n == 0:
                    stim = stim_list[target_idx]
                else:
                    stim = stimulus_sequence[i - n]
            else:
                valid_stim_found = False
                attempts = 0
                while not valid_stim_found and attempts < 100:
                    stim = rng.choice(stim_list)
                    
                    if n == 0:
                        if t_type == 'baseline' or stim != stim_list[target_idx]:
                            valid_stim_found = True
                    elif i >= n:
                        if t_type == 'baseline' or stim != stimulus_sequence[i - n]:
                            valid_stim_found = True
                    else:
                        valid_stim_found = True
                    
                    attempts += 1
                
                if not valid_stim_found:
                    print(f"Warning: Could not find valid stimulus after 100 attempts for trial {i}")
            
            stimulus_sequence.append(stim)
        
        return stimulus_sequence, trial_types
    
    
    def generate_block_order(n_back_blocks, num_blocks_per_condition, randomization_type, rng):
        """Generate block order based on randomization type"""
        num_blocks = num_blocks_per_condition * len(n_back_blocks)
        
        if randomization_type == 2:  # Full shuffle
            block_order = n_back_blocks * num_blocks_per_condition
            rng.shuffle(block_order)
            block_order = block_order[:num_blocks]
        elif randomization_type == 1:  # Shuffle initial order but repeat
            block_order = n_back_blocks.copy()
            rng.shuffle(block_order)
            block_order = block_order * num_blocks_per_condition
            block_order = block_order[:num_blocks]
        elif randomization_type == 3:  # Sorted blocks by type
            block_order = [item for item in n_back_blocks for _ in range(num_blocks_per_condition)]
        else:  # No randomization
            block_order = n_back_blocks * num_blocks_per_condition
        
        return block_order
    
    
    def generate_iti_durations(num_trials, iti_fixed, iti_jitter_min, iti_jitter_max, rng):
        """Generate ITI durations for a block"""
        iti_durations = []
        for _ in range(num_trials):
            jitter = rng.uniform(iti_jitter_min, iti_jitter_max)
            total_iti = iti_fixed + jitter
            iti_durations.append(total_iti)
        return iti_durations
    
    
    # ======================================================
    # GENERATE ALL PHASES DATA
    # ======================================================
    # This dictionary will hold ALL pre-generated data for ALL phases
    all_phases_data = {}
    
    print("=" * 60)
    print("GENERATING ALL TASK PHASES")
    print("=" * 60)
    
    for phase_idx, config in enumerate(phase_configs):
        phase_name = config['phase_name']
        print(f"\n📋 Phase {phase_idx + 1}: {config['phase_display_name']}")
        print("-" * 60)
        
        # Calculate number of blocks for this phase
        num_blocks = config['num_blocks_per_condition'] * len(config['n_back_blocks'])
        
        # Generate block order
        block_order = generate_block_order(
            config['n_back_blocks'],
            config['num_blocks_per_condition'],
            config['randomization_type'],
            rng
        )
        
        print(f"Block order: {block_order}")
        
        # Pre-generate all blocks and ITIs for this phase
        blocks_stimuli = {}
        blocks_trial_types = {}
        blocks_iti = {}
        
        for block_idx in range(num_blocks):
            block_type = block_order[block_idx]
            block_key = f"{phase_name}_block_{block_idx}"
            
            # Number of trials in this block
            num_trials_this_block = config['num_trials_per_block'] + block_type
            
            # Generate stimuli
            stimulus_sequence, trial_types = make_nback_block(
                n_back_level=block_type,
                n_trials=config['num_trials_per_block'],
                target_pct=config['target_percentage'],
                rng=rng,
                stim_list=stimulus_list,
                target_idx=config['zero_back_target_idx']
            )
            
            # Generate ITI durations
            iti_durations = generate_iti_durations(
                num_trials_this_block,
                iti_fixed_duration,
                iti_jitter_min,
                iti_jitter_max,
                rng
            )
            
            # Store
            blocks_stimuli[block_key] = stimulus_sequence
            blocks_trial_types[block_key] = trial_types
            blocks_iti[block_key] = iti_durations
            
            print(f"  Block {block_idx}: {block_type}-back, {len(stimulus_sequence)} trials, "
                  f"ITI range: {min(iti_durations):.3f}s - {max(iti_durations):.3f}s")
        
        # Store everything for this phase
        all_phases_data[phase_name] = {
            'config': config,
            'num_blocks': num_blocks,
            'block_order': block_order,
            'blocks_stimuli': blocks_stimuli,
            'blocks_trial_types': blocks_trial_types,
            'blocks_iti': blocks_iti
        }
    
    print("\n" + "=" * 60)
    print("✓ All phases generated successfully!")
    print("=" * 60)
    
    # ======================================================
    # INITIALIZE TRACKING VARIABLES
    # ======================================================
    # These will be reset at the start of each phase/block
    current_phase_idx = 0
    current_phase_name = phase_configs[0]['phase_name']
    current_block_in_phase = 0
    
    # Phase-level tracking
    phase_block_counter = 0  # Tracks blocks within current phase
    
    # Block-level tracking (reset each block)
    block_type = 0
    block_instructions = 0
    zero_back_sound_to_match_to = 0
    total_correct = 0
    block_hits = 0
    block_incorrect_rejection = 0
    block_miss_no_response_on_target = 0
    block_correct_rejections = 0
    block_false_alarms = 0
    block_no_response_on_non_target = 0
    
    # Trial-level tracking
    corrAns = None
    
    print(f"\n🎯 Starting with phase: {current_phase_name}")
    
    # Initialize phase transition message (used in Phase_Transition routine)
    phase_transition_message = ""
    # Run 'Begin Experiment' code from trigger_table
    ##TASK -ID-TRIGGER VALUES##
    # special code 100 (task start, task ID should follow immediately)
    task_start_code = 100
    #Spectal code-10s task ID for tmo muscle arturact task)
    task_ID_code = 109
    print("Starting experiment: < Auditory N-back Task >. Task ID:", task_ID_code)
    
    ##GENERAL TRIGGER VALUES##
    # special code 122 (block start)
    block_start_code = 122
    # special code 123 (block end)
    block_end_code = 123
    
    ##TASK SPECIFIC TRIGGER VALUES##
    # N.B.: only use values 1-99 and provide clear comments on used values
    audio_start_code = 10
    audio_end_code = 11
    feedback_start_code = 12
    
    
    ''' FROM NBACK FLANKER - POSSIBLY ADD FOR YOUR AS WELL
    ##TASK SPECIFIC TRIGGER VALUES##
    # N.B.: only use values 1-99 and provide clear comments on used values
    iti_start_code = 9
    letter_start_code = 10
    cue_start_code = 11
    feedback_start_code = 12
    '''
    
    # Run 'Begin Experiment' code from task_id
    dev.activate_line(bitmask=task_start_code)  # special code for task start
    core.wait(0.5)  # wait 500ms between two consecutive triggers
    dev.activate_line(bitmask=task_ID_code)  # special code for task ID
    
    
    # --- Initialize components for Routine "Information_about_block" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Arial',
        pos=(0, 0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    n_back_instructions = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='n_back_instructions',    name='n_back_instructions'
    )
    n_back_instructions.setVolume(1.0)
    sound_to_match_if_0_back = sound.Sound(
        'A', 
        secs=0.75, 
        stereo=True, 
        hamming=True, 
        speaker='sound_to_match_if_0_back',    name='sound_to_match_if_0_back'
    )
    sound_to_match_if_0_back.setVolume(1.0)
    ir_zero_otherwise_press = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='ir_zero_otherwise_press',    name='ir_zero_otherwise_press'
    )
    ir_zero_otherwise_press.setVolume(1.0)
    end_routine = keyboard.Keyboard(deviceName='end_routine')
    press_to_continue_prompt = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='press_to_continue_prompt',    name='press_to_continue_prompt'
    )
    press_to_continue_prompt.setVolume(1.0)
    force_advance = keyboard.Keyboard(deviceName='force_advance')
    practice_indicator_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='practice_indicator_sound',    name='practice_indicator_sound'
    )
    practice_indicator_sound.setVolume(1.0)
    press_to_continue = visual.TextStim(win=win, name='press_to_continue',
        text='Press right or left arrow key to continue',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    
    # --- Initialize components for Routine "Countdown" ---
    text_countdown = visual.TextStim(win=win, name='text_countdown',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    force_end_of_routine_if_pressed_5 = keyboard.Keyboard(deviceName='force_end_of_routine_if_pressed_5')
    
    # --- Initialize components for Routine "n_back_trial" ---
    # Run 'Begin Experiment' code from code_random
    # Initialize as None to distinguish from False
    corrAns = None
    Intro = visual.TextStim(win=win, name='Intro',
        text='',
        font='Arial',
        pos=(0, 0), draggable=True, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    sound_1 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='sound_1',    name='sound_1'
    )
    sound_1.setVolume(1.0)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    force_advance_2 = keyboard.Keyboard(deviceName='force_advance_2')
    text_iti_fixation = visual.TextStim(win=win, name='text_iti_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    
    # --- Initialize components for Routine "Feedback_accuracy_RT" ---
    fb = visual.TextStim(win=win, name='fb',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    rt_fb = visual.TextStim(win=win, name='rt_fb',
        text='',
        font='Open Sans',
        pos=(0, -0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "Feedback_block" ---
    # Run 'Begin Experiment' code from fb_code_block_feedback_text
    # fb_col = 'black'
    fb_2 = visual.TextStim(win=win, name='fb_2',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    force_end_of_routine_if_pressed_2 = keyboard.Keyboard(deviceName='force_end_of_routine_if_pressed_2')
    fb_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='fb_sound',    name='fb_sound'
    )
    fb_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "exp_finished" ---
    text_instr = visual.TextStim(win=win, name='text_instr',
        text='The experiment is now finished.\nThank you for participating!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    thank_you = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='thank_you',    name='thank_you'
    )
    thank_you.setVolume(1.0)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Settings" ---
    # create an object to store info about Routine Settings
    Settings = data.Routine(
        name='Settings',
        components=[],
    )
    Settings.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Settings
    Settings.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Settings.tStart = globalClock.getTime(format='float')
    Settings.status = STARTED
    thisExp.addData('Settings.started', Settings.tStart)
    Settings.maxDuration = None
    # keep track of which components have finished
    SettingsComponents = Settings.components
    for thisComponent in Settings.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Settings" ---
    Settings.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Settings.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Settings.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Settings" ---
    for thisComponent in Settings.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Settings
    Settings.tStop = globalClock.getTime(format='float')
    Settings.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Settings.stopped', Settings.tStop)
    # Run 'End Routine' code from code_exp_settings
    # Log experiment settings
    thisExp.addData('exp_setting_logged', True)
    
    # Log phase configurations
    for idx, config in enumerate(phase_configs):
        thisExp.addData(f'phase_{idx}_name', config['phase_name'])
        thisExp.addData(f'phase_{idx}_n_back_blocks', str(config['n_back_blocks']))
        thisExp.addData(f'phase_{idx}_trials_per_block', config['num_trials_per_block'])
        thisExp.addData(f'phase_{idx}_num_blocks', config['num_blocks_per_condition'])
    
    thisExp.addData('exp_num_phases', len(phase_configs))
    thisExp.addData('exp_iti_fixed', iti_fixed_duration)
    thisExp.addData('exp_iti_jitter_range', f"{iti_jitter_min}-{iti_jitter_max}")
    thisExp.nextEntry()
    # the Routine "Settings" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    phase_loop = data.TrialHandler2(
        name='phase_loop',
        nReps=len(phase_configs), 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(phase_loop)  # add the loop to the experiment
    thisPhase_loop = phase_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPhase_loop.rgb)
    if thisPhase_loop != None:
        for paramName in thisPhase_loop:
            globals()[paramName] = thisPhase_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPhase_loop in phase_loop:
        currentLoop = phase_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPhase_loop.rgb)
        if thisPhase_loop != None:
            for paramName in thisPhase_loop:
                globals()[paramName] = thisPhase_loop[paramName]
        
        # set up handler to look after randomisation of conditions etc
        block = data.TrialHandler2(
            name='block',
            nReps=all_phases_data[phase_configs[phase_loop.thisN]['phase_name']]['num_blocks'], 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(block)  # add the loop to the experiment
        thisBlock = block.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisBlock in block:
            currentLoop = block
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
            if thisBlock != None:
                for paramName in thisBlock:
                    globals()[paramName] = thisBlock[paramName]
            
            # set up handler to look after randomisation of conditions etc
            info_loop = data.TrialHandler2(
                name='info_loop',
                nReps=5.0, 
                method='sequential', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(info_loop)  # add the loop to the experiment
            thisInfo_loop = info_loop.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisInfo_loop.rgb)
            if thisInfo_loop != None:
                for paramName in thisInfo_loop:
                    globals()[paramName] = thisInfo_loop[paramName]
            
            for thisInfo_loop in info_loop:
                currentLoop = info_loop
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # abbreviate parameter names if possible (e.g. rgb = thisInfo_loop.rgb)
                if thisInfo_loop != None:
                    for paramName in thisInfo_loop:
                        globals()[paramName] = thisInfo_loop[paramName]
                
                # --- Prepare to start Routine "Information_about_block" ---
                # create an object to store info about Routine Information_about_block
                Information_about_block = data.Routine(
                    name='Information_about_block',
                    components=[text_2, n_back_instructions, sound_to_match_if_0_back, ir_zero_otherwise_press, end_routine, press_to_continue_prompt, force_advance, practice_indicator_sound, press_to_continue],
                )
                Information_about_block.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from code
                # ======================================================
                # GET CURRENT PHASE AND BLOCK DATA
                # ======================================================
                current_phase_name = phase_configs[phase_loop.thisN]['phase_name']
                current_phase_data = all_phases_data[current_phase_name]
                current_config = current_phase_data['config']
                
                # Get block information
                block_order = current_phase_data['block_order']
                block_type = block_order[block.thisN]
                n_back_condition = block_type
                n = n_back_condition
                num_trials_per_block = current_config['num_trials_per_block']
                num_trials_this_block = num_trials_per_block + n
                
                # ======================================================
                # SET PRACTICE INDICATOR AUDIO (FIXED)
                # ======================================================
                # Check if this is a practice phase using the phase_type marker
                is_practice_phase = current_config.get('phase_type', 'main') == 'practice'
                
                # Alternative check: look at phase_name patterns
                if not is_practice_phase:
                    # Also check if phase_name contains 'practice'
                    is_practice_phase = 'practice' in current_phase_name.lower()
                
                if is_practice_phase:
                    practice_indicator_audio = 'audio_instructions/this_is_practice.wav'
                    practice_indicator_duration = 2.0  # Duration of your audio file
                    print(f"  ✓ Practice mode detected - will play practice indicator")
                else:
                    practice_indicator_audio = 'audio_instructions/this_is_the_main_task.wav'
                    practice_indicator_duration = 4.0
                    print(f"  ✓ Main task mode detected - will play main task indicator")
                
                # Adjust timing for subsequent sounds
                instruction_start_time = practice_indicator_duration + 0.3  # Small gap
                
                sound_to_match_if_0_back_start_time = practice_indicator_duration + 0.3 + 7.5
                ir_zero_otherwise_press_start_time = practice_indicator_duration + 0.3 + 8.3
                press_to_continue_prompt_start_time = practice_indicator_duration + 0.3 + 11.75
                
                
                # ======================================================
                # RESET BLOCK-LEVEL TRACKING
                # ======================================================
                total_correct = 0
                block_hits = 0
                block_incorrect_rejection = 0
                block_miss_no_response_on_target = 0
                block_correct_rejections = 0
                block_false_alarms = 0
                block_no_response_on_non_target = 0
                
                # ======================================================
                # SET BLOCK-SPECIFIC INSTRUCTIONS
                # ======================================================
                block_message = f"{current_config['phase_display_name']}: {block_type}-back task"
                block_instructions = instruction_files[block_type]
                otherwise_press = otherwise_press_files[block_type]
                
                # Set zero-back target if applicable
                if block_type == 0:
                    zero_back_sound_to_match_to = stimulus_list[current_config['zero_back_target_idx']]
                else:
                    zero_back_sound_to_match_to = "audio_instructions/0_75_s_silence_48000Hz.wav"
                
                # Calculate absolute block number
                absolute_block_number = sum(all_phases_data[phase_configs[i]['phase_name']]['num_blocks'] 
                                           for i in range(phase_loop.thisN)) + block.thisN + 1
                
                print(f"\n{'='*60}")
                print(f"{'[PRACTICE] ' if is_practice_phase else '[MAIN] '}"
                      f"Phase: {current_config['phase_display_name']} | "
                      f"Block {block.thisN + 1}/{current_phase_data['num_blocks']} | "
                      f"{block_type}-back | "
                      f"{num_trials_this_block} trials")
                print(f"{'='*60}")
                
                # Verify pre-generated stimuli
                block_key = f"{current_phase_name}_block_{block.thisN}"
                print(f"Using pre-generated stimuli: {block_key}")
                
                # Timeout and response tracking
                timeout_duration = 5
                timeout_start = None
                response_made = False
                text_2.setText(block_message)
                n_back_instructions.setSound(block_instructions, hamming=True)
                n_back_instructions.setVolume(1.0, log=False)
                n_back_instructions.seek(0)
                sound_to_match_if_0_back.setSound(zero_back_sound_to_match_to, secs=0.75, hamming=True)
                sound_to_match_if_0_back.setVolume(1.0, log=False)
                sound_to_match_if_0_back.seek(0)
                ir_zero_otherwise_press.setSound(otherwise_press, secs=2.5, hamming=True)
                ir_zero_otherwise_press.setVolume(1.0, log=False)
                ir_zero_otherwise_press.seek(0)
                # create starting attributes for end_routine
                end_routine.keys = []
                end_routine.rt = []
                _end_routine_allKeys = []
                press_to_continue_prompt.setSound(press_to_continue_audio, secs=5.0, hamming=True)
                press_to_continue_prompt.setVolume(1.0, log=False)
                press_to_continue_prompt.seek(0)
                # create starting attributes for force_advance
                force_advance.keys = []
                force_advance.rt = []
                _force_advance_allKeys = []
                # Run 'Begin Routine' code from trigger_block_start_end
                if block.thisN>0:
                    # End the main experiment trial block from the previous loop
                    dev.activate_line(bitmask=block_end_code)
                    core.wait(0.5)  # wait 500ms before next block start trigger
                    
                # Start a main experiment trial block
                dev.activate_line(bitmask=block_start_code)
                # no need to wait 500ms as instructions are displayed
                practice_indicator_sound.setSound(practice_indicator_audio, hamming=True)
                practice_indicator_sound.setVolume(1.0, log=False)
                practice_indicator_sound.seek(0)
                # store start times for Information_about_block
                Information_about_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                Information_about_block.tStart = globalClock.getTime(format='float')
                Information_about_block.status = STARTED
                thisExp.addData('Information_about_block.started', Information_about_block.tStart)
                Information_about_block.maxDuration = 20
                # keep track of which components have finished
                Information_about_blockComponents = Information_about_block.components
                for thisComponent in Information_about_block.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "Information_about_block" ---
                # if trial has changed, end Routine now
                if isinstance(info_loop, data.TrialHandler2) and thisInfo_loop.thisN != info_loop.thisTrial.thisN:
                    continueRoutine = False
                Information_about_block.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # is it time to end the Routine? (based on local clock)
                    if tThisFlip > Information_about_block.maxDuration-frameTolerance:
                        Information_about_block.maxDurationReached = True
                        continueRoutine = False
                    
                    # *text_2* updates
                    
                    # if text_2 is starting this frame...
                    if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text_2.frameNStart = frameN  # exact frame index
                        text_2.tStart = t  # local t and not account for scr refresh
                        text_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_2.started')
                        # update status
                        text_2.status = STARTED
                        text_2.setAutoDraw(True)
                    
                    # if text_2 is active this frame...
                    if text_2.status == STARTED:
                        # update params
                        pass
                    
                    # if text_2 is stopping this frame...
                    if text_2.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text_2.tStartRefresh + 11.75-frameTolerance:
                            # keep track of stop time/frame for later
                            text_2.tStop = t  # not accounting for scr refresh
                            text_2.tStopRefresh = tThisFlipGlobal  # on global time
                            text_2.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_2.stopped')
                            # update status
                            text_2.status = FINISHED
                            text_2.setAutoDraw(False)
                    
                    # *n_back_instructions* updates
                    
                    # if n_back_instructions is starting this frame...
                    if n_back_instructions.status == NOT_STARTED and tThisFlip >= instruction_start_time-frameTolerance:
                        # keep track of start time/frame for later
                        n_back_instructions.frameNStart = frameN  # exact frame index
                        n_back_instructions.tStart = t  # local t and not account for scr refresh
                        n_back_instructions.tStartRefresh = tThisFlipGlobal  # on global time
                        # add timestamp to datafile
                        thisExp.addData('n_back_instructions.started', tThisFlipGlobal)
                        # update status
                        n_back_instructions.status = STARTED
                        n_back_instructions.play(when=win)  # sync with win flip
                    
                    # if n_back_instructions is stopping this frame...
                    if n_back_instructions.status == STARTED:
                        if bool(False) or n_back_instructions.isFinished:
                            # keep track of stop time/frame for later
                            n_back_instructions.tStop = t  # not accounting for scr refresh
                            n_back_instructions.tStopRefresh = tThisFlipGlobal  # on global time
                            n_back_instructions.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'n_back_instructions.stopped')
                            # update status
                            n_back_instructions.status = FINISHED
                            n_back_instructions.stop()
                    
                    # *sound_to_match_if_0_back* updates
                    
                    # if sound_to_match_if_0_back is starting this frame...
                    if sound_to_match_if_0_back.status == NOT_STARTED and tThisFlip >= sound_to_match_if_0_back_start_time-frameTolerance:
                        # keep track of start time/frame for later
                        sound_to_match_if_0_back.frameNStart = frameN  # exact frame index
                        sound_to_match_if_0_back.tStart = t  # local t and not account for scr refresh
                        sound_to_match_if_0_back.tStartRefresh = tThisFlipGlobal  # on global time
                        # add timestamp to datafile
                        thisExp.addData('sound_to_match_if_0_back.started', tThisFlipGlobal)
                        # update status
                        sound_to_match_if_0_back.status = STARTED
                        sound_to_match_if_0_back.play(when=win)  # sync with win flip
                    
                    # if sound_to_match_if_0_back is stopping this frame...
                    if sound_to_match_if_0_back.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > sound_to_match_if_0_back.tStartRefresh + 0.75-frameTolerance or sound_to_match_if_0_back.isFinished:
                            # keep track of stop time/frame for later
                            sound_to_match_if_0_back.tStop = t  # not accounting for scr refresh
                            sound_to_match_if_0_back.tStopRefresh = tThisFlipGlobal  # on global time
                            sound_to_match_if_0_back.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'sound_to_match_if_0_back.stopped')
                            # update status
                            sound_to_match_if_0_back.status = FINISHED
                            sound_to_match_if_0_back.stop()
                    
                    # *ir_zero_otherwise_press* updates
                    
                    # if ir_zero_otherwise_press is starting this frame...
                    if ir_zero_otherwise_press.status == NOT_STARTED and tThisFlip >= ir_zero_otherwise_press_start_time-frameTolerance:
                        # keep track of start time/frame for later
                        ir_zero_otherwise_press.frameNStart = frameN  # exact frame index
                        ir_zero_otherwise_press.tStart = t  # local t and not account for scr refresh
                        ir_zero_otherwise_press.tStartRefresh = tThisFlipGlobal  # on global time
                        # add timestamp to datafile
                        thisExp.addData('ir_zero_otherwise_press.started', tThisFlipGlobal)
                        # update status
                        ir_zero_otherwise_press.status = STARTED
                        ir_zero_otherwise_press.play(when=win)  # sync with win flip
                    
                    # if ir_zero_otherwise_press is stopping this frame...
                    if ir_zero_otherwise_press.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > ir_zero_otherwise_press.tStartRefresh + 2.5-frameTolerance or ir_zero_otherwise_press.isFinished:
                            # keep track of stop time/frame for later
                            ir_zero_otherwise_press.tStop = t  # not accounting for scr refresh
                            ir_zero_otherwise_press.tStopRefresh = tThisFlipGlobal  # on global time
                            ir_zero_otherwise_press.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'ir_zero_otherwise_press.stopped')
                            # update status
                            ir_zero_otherwise_press.status = FINISHED
                            ir_zero_otherwise_press.stop()
                    
                    # *end_routine* updates
                    waitOnFlip = False
                    
                    # if end_routine is starting this frame...
                    if end_routine.status == NOT_STARTED and tThisFlip >= press_to_continue_prompt_start_time-frameTolerance:
                        # keep track of start time/frame for later
                        end_routine.frameNStart = frameN  # exact frame index
                        end_routine.tStart = t  # local t and not account for scr refresh
                        end_routine.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(end_routine, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'end_routine.started')
                        # update status
                        end_routine.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(end_routine.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(end_routine.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    if end_routine.status == STARTED and not waitOnFlip:
                        theseKeys = end_routine.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=True)
                        _end_routine_allKeys.extend(theseKeys)
                        if len(_end_routine_allKeys):
                            end_routine.keys = _end_routine_allKeys[0].name  # just the first key pressed
                            end_routine.rt = _end_routine_allKeys[0].rt
                            end_routine.duration = _end_routine_allKeys[0].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # *press_to_continue_prompt* updates
                    
                    # if press_to_continue_prompt is starting this frame...
                    if press_to_continue_prompt.status == NOT_STARTED and tThisFlip >= press_to_continue_prompt_start_time-frameTolerance:
                        # keep track of start time/frame for later
                        press_to_continue_prompt.frameNStart = frameN  # exact frame index
                        press_to_continue_prompt.tStart = t  # local t and not account for scr refresh
                        press_to_continue_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                        # add timestamp to datafile
                        thisExp.addData('press_to_continue_prompt.started', tThisFlipGlobal)
                        # update status
                        press_to_continue_prompt.status = STARTED
                        press_to_continue_prompt.play(when=win)  # sync with win flip
                    
                    # if press_to_continue_prompt is stopping this frame...
                    if press_to_continue_prompt.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > press_to_continue_prompt.tStartRefresh + 5.0-frameTolerance or press_to_continue_prompt.isFinished:
                            # keep track of stop time/frame for later
                            press_to_continue_prompt.tStop = t  # not accounting for scr refresh
                            press_to_continue_prompt.tStopRefresh = tThisFlipGlobal  # on global time
                            press_to_continue_prompt.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'press_to_continue_prompt.stopped')
                            # update status
                            press_to_continue_prompt.status = FINISHED
                            press_to_continue_prompt.stop()
                    
                    # *force_advance* updates
                    waitOnFlip = False
                    
                    # if force_advance is starting this frame...
                    if force_advance.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        force_advance.frameNStart = frameN  # exact frame index
                        force_advance.tStart = t  # local t and not account for scr refresh
                        force_advance.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(force_advance, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'force_advance.started')
                        # update status
                        force_advance.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(force_advance.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(force_advance.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    if force_advance.status == STARTED and not waitOnFlip:
                        theseKeys = force_advance.getKeys(keyList=['f'], ignoreKeys=["escape"], waitRelease=False)
                        _force_advance_allKeys.extend(theseKeys)
                        if len(_force_advance_allKeys):
                            force_advance.keys = _force_advance_allKeys[-1].name  # just the last key pressed
                            force_advance.rt = _force_advance_allKeys[-1].rt
                            force_advance.duration = _force_advance_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # *practice_indicator_sound* updates
                    
                    # if practice_indicator_sound is starting this frame...
                    if practice_indicator_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        practice_indicator_sound.frameNStart = frameN  # exact frame index
                        practice_indicator_sound.tStart = t  # local t and not account for scr refresh
                        practice_indicator_sound.tStartRefresh = tThisFlipGlobal  # on global time
                        # add timestamp to datafile
                        thisExp.addData('practice_indicator_sound.started', tThisFlipGlobal)
                        # update status
                        practice_indicator_sound.status = STARTED
                        practice_indicator_sound.play(when=win)  # sync with win flip
                    
                    # if practice_indicator_sound is stopping this frame...
                    if practice_indicator_sound.status == STARTED:
                        if bool(False) or practice_indicator_sound.isFinished:
                            # keep track of stop time/frame for later
                            practice_indicator_sound.tStop = t  # not accounting for scr refresh
                            practice_indicator_sound.tStopRefresh = tThisFlipGlobal  # on global time
                            practice_indicator_sound.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'practice_indicator_sound.stopped')
                            # update status
                            practice_indicator_sound.status = FINISHED
                            practice_indicator_sound.stop()
                    
                    # *press_to_continue* updates
                    
                    # if press_to_continue is starting this frame...
                    if press_to_continue.status == NOT_STARTED and tThisFlip >= press_to_continue_prompt_start_time-frameTolerance:
                        # keep track of start time/frame for later
                        press_to_continue.frameNStart = frameN  # exact frame index
                        press_to_continue.tStart = t  # local t and not account for scr refresh
                        press_to_continue.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(press_to_continue, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'press_to_continue.started')
                        # update status
                        press_to_continue.status = STARTED
                        press_to_continue.setAutoDraw(True)
                    
                    # if press_to_continue is active this frame...
                    if press_to_continue.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[n_back_instructions, sound_to_match_if_0_back, ir_zero_otherwise_press, press_to_continue_prompt, practice_indicator_sound]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        Information_about_block.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in Information_about_block.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "Information_about_block" ---
                for thisComponent in Information_about_block.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for Information_about_block
                Information_about_block.tStop = globalClock.getTime(format='float')
                Information_about_block.tStopRefresh = tThisFlipGlobal
                thisExp.addData('Information_about_block.stopped', Information_about_block.tStop)
                # Run 'End Routine' code from code
                if 'f' in force_advance.keys:
                    info_loop.finished = True
                
                if any(k in end_routine.keys for k in ['y', 'n', 'left', 'right', 'space']):
                    info_loop.finished = True
                n_back_instructions.pause()  # ensure sound has stopped at end of Routine
                sound_to_match_if_0_back.pause()  # ensure sound has stopped at end of Routine
                ir_zero_otherwise_press.pause()  # ensure sound has stopped at end of Routine
                # check responses
                if end_routine.keys in ['', [], None]:  # No response was made
                    end_routine.keys = None
                info_loop.addData('end_routine.keys',end_routine.keys)
                if end_routine.keys != None:  # we had a response
                    info_loop.addData('end_routine.rt', end_routine.rt)
                    info_loop.addData('end_routine.duration', end_routine.duration)
                press_to_continue_prompt.pause()  # ensure sound has stopped at end of Routine
                # check responses
                if force_advance.keys in ['', [], None]:  # No response was made
                    force_advance.keys = None
                info_loop.addData('force_advance.keys',force_advance.keys)
                if force_advance.keys != None:  # we had a response
                    info_loop.addData('force_advance.rt', force_advance.rt)
                    info_loop.addData('force_advance.duration', force_advance.duration)
                practice_indicator_sound.pause()  # ensure sound has stopped at end of Routine
                # the Routine "Information_about_block" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
            # completed 5.0 repeats of 'info_loop'
            
            
            # --- Prepare to start Routine "Countdown" ---
            # create an object to store info about Routine Countdown
            Countdown = data.Routine(
                name='Countdown',
                components=[text_countdown, force_end_of_routine_if_pressed_5],
            )
            Countdown.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for force_end_of_routine_if_pressed_5
            force_end_of_routine_if_pressed_5.keys = []
            force_end_of_routine_if_pressed_5.rt = []
            _force_end_of_routine_if_pressed_5_allKeys = []
            # store start times for Countdown
            Countdown.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Countdown.tStart = globalClock.getTime(format='float')
            Countdown.status = STARTED
            thisExp.addData('Countdown.started', Countdown.tStart)
            Countdown.maxDuration = 1
            # keep track of which components have finished
            CountdownComponents = Countdown.components
            for thisComponent in Countdown.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Countdown" ---
            # if trial has changed, end Routine now
            if isinstance(block, data.TrialHandler2) and thisBlock.thisN != block.thisTrial.thisN:
                continueRoutine = False
            Countdown.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > Countdown.maxDuration-frameTolerance:
                    Countdown.maxDurationReached = True
                    continueRoutine = False
                
                # *text_countdown* updates
                
                # if text_countdown is starting this frame...
                if text_countdown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_countdown.frameNStart = frameN  # exact frame index
                    text_countdown.tStart = t  # local t and not account for scr refresh
                    text_countdown.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_countdown, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_countdown.started')
                    # update status
                    text_countdown.status = STARTED
                    text_countdown.setAutoDraw(True)
                
                # if text_countdown is active this frame...
                if text_countdown.status == STARTED:
                    # update params
                    text_countdown.setText(str(1-int(t))
                    , log=False)
                
                # if text_countdown is stopping this frame...
                if text_countdown.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_countdown.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        text_countdown.tStop = t  # not accounting for scr refresh
                        text_countdown.tStopRefresh = tThisFlipGlobal  # on global time
                        text_countdown.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_countdown.stopped')
                        # update status
                        text_countdown.status = FINISHED
                        text_countdown.setAutoDraw(False)
                
                # *force_end_of_routine_if_pressed_5* updates
                waitOnFlip = False
                
                # if force_end_of_routine_if_pressed_5 is starting this frame...
                if force_end_of_routine_if_pressed_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    force_end_of_routine_if_pressed_5.frameNStart = frameN  # exact frame index
                    force_end_of_routine_if_pressed_5.tStart = t  # local t and not account for scr refresh
                    force_end_of_routine_if_pressed_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(force_end_of_routine_if_pressed_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'force_end_of_routine_if_pressed_5.started')
                    # update status
                    force_end_of_routine_if_pressed_5.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(force_end_of_routine_if_pressed_5.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(force_end_of_routine_if_pressed_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if force_end_of_routine_if_pressed_5.status == STARTED and not waitOnFlip:
                    theseKeys = force_end_of_routine_if_pressed_5.getKeys(keyList=['f'], ignoreKeys=["escape"], waitRelease=False)
                    _force_end_of_routine_if_pressed_5_allKeys.extend(theseKeys)
                    if len(_force_end_of_routine_if_pressed_5_allKeys):
                        force_end_of_routine_if_pressed_5.keys = _force_end_of_routine_if_pressed_5_allKeys[-1].name  # just the last key pressed
                        force_end_of_routine_if_pressed_5.rt = _force_end_of_routine_if_pressed_5_allKeys[-1].rt
                        force_end_of_routine_if_pressed_5.duration = _force_end_of_routine_if_pressed_5_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Countdown.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Countdown.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Countdown" ---
            for thisComponent in Countdown.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Countdown
            Countdown.tStop = globalClock.getTime(format='float')
            Countdown.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Countdown.stopped', Countdown.tStop)
            # check responses
            if force_end_of_routine_if_pressed_5.keys in ['', [], None]:  # No response was made
                force_end_of_routine_if_pressed_5.keys = None
            block.addData('force_end_of_routine_if_pressed_5.keys',force_end_of_routine_if_pressed_5.keys)
            if force_end_of_routine_if_pressed_5.keys != None:  # we had a response
                block.addData('force_end_of_routine_if_pressed_5.rt', force_end_of_routine_if_pressed_5.rt)
                block.addData('force_end_of_routine_if_pressed_5.duration', force_end_of_routine_if_pressed_5.duration)
            # the Routine "Countdown" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            trials = data.TrialHandler2(
                name='trials',
                nReps=len(all_phases_data[phase_configs[phase_loop.thisN]['phase_name']]['blocks_stimuli'][f"{phase_configs[phase_loop.thisN]['phase_name']}_block_{block.thisN}"]), 
                method='sequential', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(trials)  # add the loop to the experiment
            thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisTrial in trials:
                currentLoop = trials
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
                if thisTrial != None:
                    for paramName in thisTrial:
                        globals()[paramName] = thisTrial[paramName]
                
                # --- Prepare to start Routine "n_back_trial" ---
                # create an object to store info about Routine n_back_trial
                n_back_trial = data.Routine(
                    name='n_back_trial',
                    components=[Intro, sound_1, key_resp, force_advance_2, text_iti_fixation],
                )
                n_back_trial.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from code_random
                # ======================================================
                # GET CURRENT PHASE AND TRIAL DATA
                # ======================================================
                current_phase_name = phase_configs[phase_loop.thisN]['phase_name']
                current_phase_data = all_phases_data[current_phase_name]
                current_config = current_phase_data['config']
                
                # Retrieve pre-generated stimuli for this block
                block_key = f"{current_phase_name}_block_{block.thisN}"
                stimulus_sequence = current_phase_data['blocks_stimuli'][block_key]
                trial_types = current_phase_data['blocks_trial_types'][block_key]
                
                # Get current trial's values
                current_stimulus = stimulus_sequence[trials.thisN]
                trial_type = trial_types[trials.thisN]
                
                # Get n-back level for this block
                block_order = current_phase_data['block_order']
                n = block_order[block.thisN]
                
                # Determine if this is a baseline trial
                is_baseline_trial = trials.thisN < n
                
                print(f"  Trial {trials.thisN + 1}/{len(stimulus_sequence)}: "
                      f"Stimulus={current_stimulus.split('_')[3]}, "
                      f"Type={trial_type}, "
                      f"Baseline={is_baseline_trial}")
                Intro.setColor([0.9608, 0.8431, 0.6863], colorSpace='rgb')
                Intro.setText('Sound')
                sound_1.setSound(current_stimulus, hamming=True)
                sound_1.setVolume(1.0, log=False)
                sound_1.seek(0)
                # create starting attributes for key_resp
                key_resp.keys = []
                key_resp.rt = []
                _key_resp_allKeys = []
                # create starting attributes for force_advance_2
                force_advance_2.keys = []
                force_advance_2.rt = []
                _force_advance_2_allKeys = []
                # Run 'Begin Routine' code from trigger_trial
                audio_trigger_started = False
                audio_end_started = False
                # Run 'Begin Routine' code from code_iti_timing
                # Get phase and block info
                current_phase_name = phase_configs[phase_loop.thisN]['phase_name']
                current_phase_data = all_phases_data[current_phase_name]
                
                # Get ITI for this trial
                block_key = f"{current_phase_name}_block_{block.thisN}"
                current_iti_duration = current_phase_data['blocks_iti'][block_key][trials.thisN]
                
                # Sound duration
                stimulus_duration = 0.75
                
                # Total trial duration
                total_trial_duration = stimulus_duration + current_iti_duration
                iti_start_time = stimulus_duration
                
                print(f"    Stimulus={stimulus_duration}s, ITI={current_iti_duration:.3f}s, Total={total_trial_duration:.3f}s")
                # store start times for n_back_trial
                n_back_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                n_back_trial.tStart = globalClock.getTime(format='float')
                n_back_trial.status = STARTED
                thisExp.addData('n_back_trial.started', n_back_trial.tStart)
                n_back_trial.maxDuration = total_trial_duration
                # keep track of which components have finished
                n_back_trialComponents = n_back_trial.components
                for thisComponent in n_back_trial.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "n_back_trial" ---
                # if trial has changed, end Routine now
                if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                    continueRoutine = False
                n_back_trial.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # is it time to end the Routine? (based on local clock)
                    if tThisFlip > n_back_trial.maxDuration-frameTolerance:
                        n_back_trial.maxDurationReached = True
                        continueRoutine = False
                    
                    # *Intro* updates
                    
                    # if Intro is starting this frame...
                    if Intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Intro.frameNStart = frameN  # exact frame index
                        Intro.tStart = t  # local t and not account for scr refresh
                        Intro.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Intro, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Intro.started')
                        # update status
                        Intro.status = STARTED
                        Intro.setAutoDraw(True)
                    
                    # if Intro is active this frame...
                    if Intro.status == STARTED:
                        # update params
                        pass
                    
                    # if Intro is stopping this frame...
                    if Intro.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > Intro.tStartRefresh + stimulus_duration-frameTolerance:
                            # keep track of stop time/frame for later
                            Intro.tStop = t  # not accounting for scr refresh
                            Intro.tStopRefresh = tThisFlipGlobal  # on global time
                            Intro.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'Intro.stopped')
                            # update status
                            Intro.status = FINISHED
                            Intro.setAutoDraw(False)
                    
                    # *sound_1* updates
                    
                    # if sound_1 is starting this frame...
                    if sound_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        sound_1.frameNStart = frameN  # exact frame index
                        sound_1.tStart = t  # local t and not account for scr refresh
                        sound_1.tStartRefresh = tThisFlipGlobal  # on global time
                        # add timestamp to datafile
                        thisExp.addData('sound_1.started', tThisFlipGlobal)
                        # update status
                        sound_1.status = STARTED
                        sound_1.play(when=win)  # sync with win flip
                    
                    # if sound_1 is stopping this frame...
                    if sound_1.status == STARTED:
                        if bool(False) or sound_1.isFinished:
                            # keep track of stop time/frame for later
                            sound_1.tStop = t  # not accounting for scr refresh
                            sound_1.tStopRefresh = tThisFlipGlobal  # on global time
                            sound_1.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'sound_1.stopped')
                            # update status
                            sound_1.status = FINISHED
                            sound_1.stop()
                    
                    # *key_resp* updates
                    waitOnFlip = False
                    
                    # if key_resp is starting this frame...
                    if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp.frameNStart = frameN  # exact frame index
                        key_resp.tStart = t  # local t and not account for scr refresh
                        key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp.started')
                        # update status
                        key_resp.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    if key_resp.status == STARTED and not waitOnFlip:
                        theseKeys = key_resp.getKeys(keyList=['y','n','left','right'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_allKeys.extend(theseKeys)
                        if len(_key_resp_allKeys):
                            key_resp.keys = [key.name for key in _key_resp_allKeys]  # storing all keys
                            key_resp.rt = [key.rt for key in _key_resp_allKeys]
                            key_resp.duration = [key.duration for key in _key_resp_allKeys]
                            # was this correct?
                            if (key_resp.keys == str(corrAns)) or (key_resp.keys == corrAns):
                                key_resp.corr = 1
                            else:
                                key_resp.corr = 0
                    
                    # *force_advance_2* updates
                    waitOnFlip = False
                    
                    # if force_advance_2 is starting this frame...
                    if force_advance_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        force_advance_2.frameNStart = frameN  # exact frame index
                        force_advance_2.tStart = t  # local t and not account for scr refresh
                        force_advance_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(force_advance_2, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'force_advance_2.started')
                        # update status
                        force_advance_2.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(force_advance_2.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(force_advance_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if force_advance_2 is stopping this frame...
                    if force_advance_2.status == STARTED:
                        # is it time to stop? (based on local clock)
                        if tThisFlip > 14.75-frameTolerance:
                            # keep track of stop time/frame for later
                            force_advance_2.tStop = t  # not accounting for scr refresh
                            force_advance_2.tStopRefresh = tThisFlipGlobal  # on global time
                            force_advance_2.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'force_advance_2.stopped')
                            # update status
                            force_advance_2.status = FINISHED
                            force_advance_2.status = FINISHED
                    if force_advance_2.status == STARTED and not waitOnFlip:
                        theseKeys = force_advance_2.getKeys(keyList=['f'], ignoreKeys=["escape"], waitRelease=False)
                        _force_advance_2_allKeys.extend(theseKeys)
                        if len(_force_advance_2_allKeys):
                            force_advance_2.keys = _force_advance_2_allKeys[-1].name  # just the last key pressed
                            force_advance_2.rt = _force_advance_2_allKeys[-1].rt
                            force_advance_2.duration = _force_advance_2_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    # Run 'Each Frame' code from trigger_trial
                    if sound_1.status == STARTED and not audio_trigger_started:
                        win.callOnFlip(dev.activate_line, bitmask=audio_start_code)
                        audio_trigger_started = True
                        thisExp.addData('audio_trigger_sent', True)  # Log trigger was sent
                    
                    if sound_1.status == FINISHED and not audio_end_started:
                        win.callOnFlip(dev.activate_line, bitmask=audio_end_code)
                        audio_end_started = True
                        thisExp.addData('audio_end_trigger_sent', True)
                        
                        
                    ''' 
                    Adapted from Alex's n-back/flanker code so might not need the second trigger
                    '''
                    
                    
                    # *text_iti_fixation* updates
                    
                    # if text_iti_fixation is starting this frame...
                    if text_iti_fixation.status == NOT_STARTED and tThisFlip >= iti_start_time-frameTolerance:
                        # keep track of start time/frame for later
                        text_iti_fixation.frameNStart = frameN  # exact frame index
                        text_iti_fixation.tStart = t  # local t and not account for scr refresh
                        text_iti_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_iti_fixation, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_iti_fixation.started')
                        # update status
                        text_iti_fixation.status = STARTED
                        text_iti_fixation.setAutoDraw(True)
                    
                    # if text_iti_fixation is active this frame...
                    if text_iti_fixation.status == STARTED:
                        # update params
                        pass
                    
                    # if text_iti_fixation is stopping this frame...
                    if text_iti_fixation.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text_iti_fixation.tStartRefresh + current_iti_duration-frameTolerance:
                            # keep track of stop time/frame for later
                            text_iti_fixation.tStop = t  # not accounting for scr refresh
                            text_iti_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                            text_iti_fixation.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_iti_fixation.stopped')
                            # update status
                            text_iti_fixation.status = FINISHED
                            text_iti_fixation.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[sound_1]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        n_back_trial.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in n_back_trial.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "n_back_trial" ---
                for thisComponent in n_back_trial.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for n_back_trial
                n_back_trial.tStop = globalClock.getTime(format='float')
                n_back_trial.tStopRefresh = tThisFlipGlobal
                thisExp.addData('n_back_trial.stopped', n_back_trial.tStop)
                # Run 'End Routine' code from code_random
                # ======================================================
                # EXTRACT RESPONSE DATA
                # ======================================================
                keys = key_resp.keys
                
                if keys:
                    if isinstance(keys, list):
                        response_key = keys[0] if len(keys) > 0 else None
                        rt = key_resp.rt[0] if key_resp.rt and len(key_resp.rt) > 0 else None
                        num_responses = len(keys)
                    else:
                        response_key = keys
                        rt = key_resp.rt
                        num_responses = 1
                else:
                    response_key = None
                    rt = None
                    num_responses = 0
                
                # ======================================================
                # CATEGORIZE RESPONSE TYPE
                # ======================================================
                if response_key:
                    if response_key in ['y', 'right', 'r']:
                        response_type = 'target_response'
                    elif response_key in ['n', 'left', 'l']:
                        response_type = 'non_target_response'
                    else:
                        response_type = 'invalid_response'
                else:
                    response_type = 'no_response'
                
                # ======================================================
                # INITIALIZE PERFORMANCE METRICS
                # ======================================================
                is_hit = None
                is_incorrect_rejection = None
                is_miss_no_response_on_target = None
                is_correct_rejection = None
                is_false_alarm = None
                is_no_response_on_non_target = None
                
                # ======================================================
                # CALCULATE CORRECTNESS AND SDT METRICS
                # ======================================================
                if trials.thisN < n:
                    # Baseline trials - not scored
                    corrAns = None
                    performance_category = 'baseline'
                else:
                    # Scorable trials - apply Signal Detection Theory
                    if trial_type == 'target':
                        if response_type == 'target_response':
                            corrAns = True
                            performance_category = 'hit'
                            is_hit = 1
                            is_incorrect_rejection = 0
                            is_miss_no_response_on_target = 0
                            is_correct_rejection = 0
                            is_false_alarm = 0
                            is_no_response_on_non_target = 0
                            block_hits += 1
                        elif response_type == 'non_target_response':
                            corrAns = False
                            performance_category = 'incorrect_rejection'
                            is_hit = 0
                            is_incorrect_rejection = 1
                            is_miss_no_response_on_target = 0
                            is_correct_rejection = 0
                            is_false_alarm = 0
                            is_no_response_on_non_target = 0
                            block_incorrect_rejection += 1
                        else:
                            corrAns = False
                            performance_category = 'miss_no_response_on_target'
                            is_hit = 0
                            is_incorrect_rejection = 0
                            is_miss_no_response_on_target = 1
                            is_correct_rejection = 0
                            is_false_alarm = 0
                            is_no_response_on_non_target = 0
                            block_miss_no_response_on_target += 1
                    else:  # non-target
                        if response_type == 'non_target_response':
                            corrAns = True
                            performance_category = 'correct_rejection'
                            is_hit = 0
                            is_incorrect_rejection = 0
                            is_miss_no_response_on_target = 0
                            is_correct_rejection = 1
                            is_false_alarm = 0
                            is_no_response_on_non_target = 0
                            block_correct_rejections += 1
                        elif response_type == 'target_response':
                            corrAns = False
                            performance_category = 'false_alarm'
                            is_hit = 0
                            is_incorrect_rejection = 0
                            is_miss_no_response_on_target = 0
                            is_correct_rejection = 0
                            is_false_alarm = 1
                            is_no_response_on_non_target = 0
                            block_false_alarms += 1
                        else:
                            corrAns = False
                            performance_category = 'no_response_on_non_target'
                            is_hit = 0
                            is_incorrect_rejection = 0
                            is_miss_no_response_on_target = 0
                            is_correct_rejection = 0
                            is_false_alarm = 0
                            is_no_response_on_non_target = 1
                            block_no_response_on_non_target += 1
                    
                    # Update running total
                    if corrAns:
                        total_correct += 1
                
                # ======================================================
                # CALCULATE TRIAL POSITION METRICS
                # ======================================================
                trial_position_in_block = trials.thisN + 1
                scorable_trial_number = (trials.thisN + 1 - n) if trials.thisN >= n else 0
                
                # Determine target stimulus for verification
                if n == 0:
                    target_stimulus = stimulus_list[current_config['zero_back_target_idx']]
                elif trials.thisN >= n:
                    target_stimulus = stimulus_sequence[trials.thisN - n]
                else:
                    target_stimulus = 'N/A'
                
                # Check if current stimulus matches target (ground truth)
                if trials.thisN >= n:
                    if n == 0:
                        is_match_trial = (current_stimulus == stimulus_list[current_config['zero_back_target_idx']])
                    else:
                        is_match_trial = (current_stimulus == stimulus_sequence[trials.thisN - n])
                else:
                    is_match_trial = None
                
                # ======================================================
                # LOG ALL DATA TO CSV
                # ======================================================
                
                # Phase information
                thisExp.addData('phase_name', current_phase_name)
                thisExp.addData('phase_display_name', current_config['phase_display_name'])
                thisExp.addData('phase_number', phase_loop.thisN + 1)
                thisExp.addData('is_practice', 1 if current_phase_name == 'practice' else 0)
                
                # Participant and session
                thisExp.addData('participant_id', expInfo['participant'])
                thisExp.addData('session', expInfo.get('session', '001'))
                thisExp.addData('date', expInfo['date'])
                
                # Block-level information
                absolute_block_number = sum(all_phases_data[phase_configs[i]['phase_name']]['num_blocks'] 
                                           for i in range(phase_loop.thisN)) + block.thisN + 1
                thisExp.addData('absolute_block_number', absolute_block_number)
                thisExp.addData('block_number_in_phase', block.thisN + 1)
                thisExp.addData('n_back_level', n)
                thisExp.addData('block_type', f'{n}_back')
                thisExp.addData('trials_per_block', current_config['num_trials_per_block'])
                thisExp.addData('total_blocks_in_phase', current_phase_data['num_blocks'])
                
                # Trial-level information
                thisExp.addData('trial_number_overall', trials.thisN + 1)
                thisExp.addData('trial_number_in_block', trial_position_in_block)
                thisExp.addData('scorable_trial_number', scorable_trial_number)
                thisExp.addData('is_baseline_trial', 1 if trials.thisN < n else 0)
                
                # Stimulus information
                thisExp.addData('current_stimulus', current_stimulus)
                thisExp.addData('target_stimulus', target_stimulus)
                stimulus_freq = current_stimulus.split('_')[2].replace('frequency', '') if 'frequency' in current_stimulus else 'N/A'
                thisExp.addData('stimulus_frequency_hz', stimulus_freq)
                
                # Trial design
                thisExp.addData('trial_type', trial_type)
                thisExp.addData('is_match_trial', is_match_trial)
                
                # Response information
                thisExp.addData('response_key', response_key if response_key else 'none')
                thisExp.addData('response_type', response_type)
                thisExp.addData('reaction_time_ms', round(rt * 1000, 2) if rt else None)
                thisExp.addData('reaction_time_s', round(rt, 4) if rt else None)
                
                # Accuracy and performance
                thisExp.addData('correct', 1 if corrAns == True else (0 if corrAns == False else 'N/A'))
                thisExp.addData('accuracy_binary', corrAns if corrAns is not None else 'N/A')
                thisExp.addData('performance_category', performance_category)
                
                # SDT metrics
                thisExp.addData('hit', is_hit if is_hit is not None else 'N/A')
                thisExp.addData('incorrect_rejection', is_incorrect_rejection if is_incorrect_rejection is not None else 'N/A')
                thisExp.addData('miss_no_response_on_target', is_miss_no_response_on_target if is_miss_no_response_on_target is not None else 'N/A')
                thisExp.addData('correct_rejection', is_correct_rejection if is_correct_rejection is not None else 'N/A')
                thisExp.addData('false_alarm', is_false_alarm if is_false_alarm is not None else 'N/A')
                thisExp.addData('no_response_on_non_target', is_no_response_on_non_target if is_no_response_on_non_target is not None else 'N/A')
                
                # Running performance
                scorable_trials_so_far = max(0, trials.thisN + 1 - n)
                if scorable_trials_so_far > 0:
                    running_accuracy = total_correct / scorable_trials_so_far
                    thisExp.addData('running_accuracy', round(running_accuracy, 4))
                    thisExp.addData('total_correct_so_far', total_correct)
                    thisExp.addData('scorable_trials_so_far', scorable_trials_so_far)
                else:
                    thisExp.addData('running_accuracy', 'N/A')
                    thisExp.addData('total_correct_so_far', 0)
                    thisExp.addData('scorable_trials_so_far', 0)
                
                # Timing
                thisExp.addData('trial_onset_time', t)
                
                # Context (previous stimuli)
                if trials.thisN >= 1:
                    prev_stim_1 = stimulus_sequence[trials.thisN - 1]
                    prev_freq_1 = prev_stim_1.split('_')[3].replace('frequency', '') if 'frequency' in prev_stim_1 else 'N/A'
                    thisExp.addData('previous_stimulus_n1', prev_stim_1)
                    thisExp.addData('previous_frequency_n1', prev_freq_1)
                else:
                    thisExp.addData('previous_stimulus_n1', 'N/A')
                    thisExp.addData('previous_frequency_n1', 'N/A')
                
                if trials.thisN >= 2:
                    prev_stim_2 = stimulus_sequence[trials.thisN - 2]
                    prev_freq_2 = prev_stim_2.split('_')[3].replace('frequency', '') if 'frequency' in prev_stim_2 else 'N/A'
                    thisExp.addData('previous_stimulus_n2', prev_stim_2)
                    thisExp.addData('previous_frequency_n2', prev_freq_2)
                else:
                    thisExp.addData('previous_stimulus_n2', 'N/A')
                    thisExp.addData('previous_frequency_n2', 'N/A')
                
                if trials.thisN >= 3:
                    prev_stim_3 = stimulus_sequence[trials.thisN - 3]
                    prev_freq_3 = prev_stim_3.split('_')[3].replace('frequency', '') if 'frequency' in prev_stim_3 else 'N/A'
                    thisExp.addData('previous_stimulus_n3', prev_stim_3)
                    thisExp.addData('previous_frequency_n3', prev_freq_3)
                else:
                    thisExp.addData('previous_stimulus_n3', 'N/A')
                    thisExp.addData('previous_frequency_n3', 'N/A')
                
                # Update key_resp.corr for PsychoPy
                if corrAns is None:
                    key_resp.corr = -1
                else:
                    key_resp.corr = int(corrAns)
                
                # Multiple responses
                thisExp.addData('num_responses', num_responses)
                if num_responses > 1:
                    thisExp.addData('all_response_keys', str(keys))
                    thisExp.addData('all_response_times', str([round(t * 1000, 2) for t in key_resp.rt]))
                else:
                    thisExp.addData('all_response_keys', response_key if response_key else 'none')
                    thisExp.addData('all_response_times', round(rt * 1000, 2) if rt else None)
                
                # ======================================================
                # LOG ITI INFORMATION
                # ======================================================
                thisExp.addData('iti_duration', round(current_iti_duration, 4))
                thisExp.addData('iti_fixed_component', iti_fixed_duration)
                thisExp.addData('iti_jitter_component', round(current_iti_duration - iti_fixed_duration, 4))
                thisExp.addData('stimulus_duration', stimulus_duration)
                thisExp.addData('total_trial_duration', round(total_trial_duration, 4))
                
                # Response timing analysis
                if rt is not None:
                    if rt <= stimulus_duration:
                        response_period = 'during_stimulus'
                    else:
                        response_period = 'during_iti'
                        time_into_iti = rt - stimulus_duration
                        thisExp.addData('response_time_into_iti', round(time_into_iti, 4))
                    
                    thisExp.addData('response_period', response_period)
                else:
                    thisExp.addData('response_period', 'no_response')
                    thisExp.addData('response_time_into_iti', 'N/A')
                
                # Console output
                # Format RT and accuracy for printing
                rt_str = f"{rt:.3f}" if rt else "N/A"
                acc_str = f"{running_accuracy:.2%}" if scorable_trials_so_far > 0 else "N/A"
                
                print(f"    {performance_category} | RT: {rt_str} | Acc: {acc_str}")
                sound_1.pause()  # ensure sound has stopped at end of Routine
                # check responses
                if key_resp.keys in ['', [], None]:  # No response was made
                    key_resp.keys = None
                    # was no response the correct answer?!
                    if str(corrAns).lower() == 'none':
                       key_resp.corr = 1;  # correct non-response
                    else:
                       key_resp.corr = 0;  # failed to respond (incorrectly)
                # store data for trials (TrialHandler)
                trials.addData('key_resp.keys',key_resp.keys)
                trials.addData('key_resp.corr', key_resp.corr)
                if key_resp.keys != None:  # we had a response
                    trials.addData('key_resp.rt', key_resp.rt)
                    trials.addData('key_resp.duration', key_resp.duration)
                # check responses
                if force_advance_2.keys in ['', [], None]:  # No response was made
                    force_advance_2.keys = None
                trials.addData('force_advance_2.keys',force_advance_2.keys)
                if force_advance_2.keys != None:  # we had a response
                    trials.addData('force_advance_2.rt', force_advance_2.rt)
                    trials.addData('force_advance_2.duration', force_advance_2.duration)
                # the Routine "n_back_trial" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "Feedback_accuracy_RT" ---
                # create an object to store info about Routine Feedback_accuracy_RT
                Feedback_accuracy_RT = data.Routine(
                    name='Feedback_accuracy_RT',
                    components=[fb, rt_fb],
                )
                Feedback_accuracy_RT.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from fb_code
                # Provide feedback based on the previous trial
                fb_text = 'no key_resp component found - check console'
                fb_col = 'black'
                
                try:
                    # Check if this was a baseline trial
                    if trials.thisN < n:
                        fb_text = 'Baseline trial'
                        fb_col = 'blue'
                    elif key_resp.corr == 1:
                        fb_text = 'Correct!'
                        fb_col = 'green'
                    elif key_resp.corr == 0:
                        fb_text = 'Incorrect'
                        fb_col = 'red'
                    else:
                        fb_text = 'No response recorded'
                        fb_col = 'orange'
                
                except Exception as e:
                    print(f'Feedback error: {e}')
                    print('Make sure key_resp component exists and "Store Correct" is enabled')
                fb.setColor(fb_col, colorSpace='rgb')
                fb.setText(fb_text)
                # Run 'Begin Routine' code from rt_fb_code
                # Provide RT feedback that works with "Store All Keys" mode
                fb_text = 'No key press detected. Please press left or right arrow.'
                
                try:
                    rt = key_resp.rt
                    # Handle both single value and list
                    if rt:
                        if isinstance(rt, list):
                            rt_value = rt[0]  # Use first response time
                        else:
                            rt_value = rt
                        fb_text = 'RT: ' + str(round(rt_value * 1000) / 1000) + ' seconds'
                except:
                    print('Make sure that you have:\n1. a routine with a keyboard component in it called "key_resp"\n2. that data is set to store from first or last key')
                rt_fb.setText(fb_text)
                # Run 'Begin Routine' code from trigger_fb
                feedback_trigger_started = False
                
                # store start times for Feedback_accuracy_RT
                Feedback_accuracy_RT.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                Feedback_accuracy_RT.tStart = globalClock.getTime(format='float')
                Feedback_accuracy_RT.status = STARTED
                thisExp.addData('Feedback_accuracy_RT.started', Feedback_accuracy_RT.tStart)
                Feedback_accuracy_RT.maxDuration = 0.25
                # keep track of which components have finished
                Feedback_accuracy_RTComponents = Feedback_accuracy_RT.components
                for thisComponent in Feedback_accuracy_RT.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "Feedback_accuracy_RT" ---
                # if trial has changed, end Routine now
                if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                    continueRoutine = False
                Feedback_accuracy_RT.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # is it time to end the Routine? (based on local clock)
                    if tThisFlip > Feedback_accuracy_RT.maxDuration-frameTolerance:
                        Feedback_accuracy_RT.maxDurationReached = True
                        continueRoutine = False
                    
                    # *fb* updates
                    
                    # if fb is starting this frame...
                    if fb.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        fb.frameNStart = frameN  # exact frame index
                        fb.tStart = t  # local t and not account for scr refresh
                        fb.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(fb, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fb.started')
                        # update status
                        fb.status = STARTED
                        fb.setAutoDraw(True)
                    
                    # if fb is active this frame...
                    if fb.status == STARTED:
                        # update params
                        pass
                    
                    # *rt_fb* updates
                    
                    # if rt_fb is starting this frame...
                    if rt_fb.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        rt_fb.frameNStart = frameN  # exact frame index
                        rt_fb.tStart = t  # local t and not account for scr refresh
                        rt_fb.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(rt_fb, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'rt_fb.started')
                        # update status
                        rt_fb.status = STARTED
                        rt_fb.setAutoDraw(True)
                    
                    # if rt_fb is active this frame...
                    if rt_fb.status == STARTED:
                        # update params
                        pass
                    # Run 'Each Frame' code from trigger_fb
                    if fb.status == STARTED and not feedback_trigger_started:
                        win.callOnFlip(dev.activate_line, bitmask=feedback_start_code)
                        feedback_trigger_started = True
                    
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        Feedback_accuracy_RT.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in Feedback_accuracy_RT.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "Feedback_accuracy_RT" ---
                for thisComponent in Feedback_accuracy_RT.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for Feedback_accuracy_RT
                Feedback_accuracy_RT.tStop = globalClock.getTime(format='float')
                Feedback_accuracy_RT.tStopRefresh = tThisFlipGlobal
                thisExp.addData('Feedback_accuracy_RT.stopped', Feedback_accuracy_RT.tStop)
                # the Routine "Feedback_accuracy_RT" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
            # completed len(all_phases_data[phase_configs[phase_loop.thisN]['phase_name']]['blocks_stimuli'][f"{phase_configs[phase_loop.thisN]['phase_name']}_block_{block.thisN}"]) repeats of 'trials'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            # --- Prepare to start Routine "Feedback_block" ---
            # create an object to store info about Routine Feedback_block
            Feedback_block = data.Routine(
                name='Feedback_block',
                components=[fb_2, force_end_of_routine_if_pressed_2, fb_sound],
            )
            Feedback_block.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from fb_code_block_feedback_text
            # Get phase info
            current_phase_name = phase_configs[phase_loop.thisN]['phase_name']
            current_config = all_phases_data[current_phase_name]['config']
            num_trials_per_block = current_config['num_trials_per_block']
            
            fb_col = 'black'
            
            if num_trials_this_block > n:
                total_trials = num_trials_per_block
                percent_correct = total_correct / total_trials
                
                phase_label = current_config['phase_display_name']
                fb_code_block_feedback_text = (
                    f"{phase_label} Block Complete\n"
                    f"Accuracy: {round(percent_correct * 100, 1)}%"
                )
                
                # Determine feedback audio
                if percent_correct >= 0.9:
                    feedback_sound = 'audio_instructions/performance_5_excellent.wav'
                elif percent_correct >= 0.8:
                    feedback_sound = "audio_instructions/performance_3_quite_good.wav"
                elif percent_correct >= 0.6:
                    feedback_sound = "audio_instructions/performance_2_okay.wav"
                else:
                    feedback_sound = 'audio_instructions/performance_1_not_so_good.wav'
            else:
                fb_code_block_feedback_text = 'Not enough trials'
                percent_correct = 0
                feedback_sound = 'audio_instructions/performance_1_not_so_good.wav'
            
            # Calculate total misses
            block_total_misses = block_incorrect_rejection + block_miss_no_response_on_target
            
            # ======================================================
            # LOG BLOCK SUMMARY DATA
            # ======================================================
            absolute_block_number = sum(all_phases_data[phase_configs[i]['phase_name']]['num_blocks'] 
                                       for i in range(phase_loop.thisN)) + block.thisN + 1
            
            thisExp.addData('phase_name', current_phase_name)
            thisExp.addData('phase_display_name', current_config['phase_display_name'])
            thisExp.addData('is_practice', 1 if current_phase_name == 'practice' else 0)
            thisExp.addData('block_summary', True)
            thisExp.addData('absolute_block_number', absolute_block_number)
            thisExp.addData('block_number_in_phase', block.thisN + 1)
            thisExp.addData('block_final_accuracy', round(percent_correct, 4))
            thisExp.addData('block_n_back_level', n)
            thisExp.addData('block_total_scorable_trials', total_trials if num_trials_this_block > n else 0)
            thisExp.addData('block_total_correct', total_correct)
            thisExp.addData('block_total_hits', block_hits)
            thisExp.addData('block_total_incorrect_rejection', block_incorrect_rejection)
            thisExp.addData('block_total_miss_no_response_on_target', block_miss_no_response_on_target)
            thisExp.addData('block_total_misses', block_total_misses)
            thisExp.addData('block_total_false_alarms', block_false_alarms)
            thisExp.addData('block_total_correct_rejections', block_correct_rejections)
            thisExp.addData('block_total_no_response_on_non_target', block_no_response_on_non_target)
            
            # SDT metrics
            if block_hits + block_total_misses > 0:
                hit_rate = block_hits / (block_hits + block_total_misses)
                thisExp.addData('block_hit_rate', round(hit_rate, 4))
            else:
                thisExp.addData('block_hit_rate', 'N/A')
            
            if block_false_alarms + block_correct_rejections > 0:
                fa_rate = block_false_alarms / (block_false_alarms + block_correct_rejections)
                thisExp.addData('block_false_alarm_rate', round(fa_rate, 4))
            else:
                thisExp.addData('block_false_alarm_rate', 'N/A')
            
            # d-prime
            if 'hit_rate' in locals() and 'fa_rate' in locals() and hit_rate != 'N/A' and fa_rate != 'N/A':
                hit_rate_adj = min(max(hit_rate, 0.01), 0.99)
                fa_rate_adj = min(max(fa_rate, 0.01), 0.99)
                
                from scipy.stats import norm
                z_hit = norm.ppf(hit_rate_adj)
                z_fa = norm.ppf(fa_rate_adj)
                d_prime = z_hit - z_fa
                
                thisExp.addData('block_d_prime', round(d_prime, 4))
            else:
                thisExp.addData('block_d_prime', 'N/A')
            
            print(f"\n{current_config['phase_display_name']} Block {block.thisN + 1} Summary:")
            print(f"  Accuracy: {percent_correct:.2%}")
            print(f"  Hits: {block_hits}, IR: {block_incorrect_rejection}, Miss: {block_miss_no_response_on_target}")
            print(f"  FA: {block_false_alarms}, CR: {block_correct_rejections}, No resp (NT): {block_no_response_on_non_target}")
            fb_2.setColor(fb_col, colorSpace='rgb')
            fb_2.setText(fb_code_block_feedback_text)
            # create starting attributes for force_end_of_routine_if_pressed_2
            force_end_of_routine_if_pressed_2.keys = []
            force_end_of_routine_if_pressed_2.rt = []
            _force_end_of_routine_if_pressed_2_allKeys = []
            fb_sound.setSound(feedback_sound, hamming=True)
            fb_sound.setVolume(1.0, log=False)
            fb_sound.seek(0)
            # store start times for Feedback_block
            Feedback_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Feedback_block.tStart = globalClock.getTime(format='float')
            Feedback_block.status = STARTED
            thisExp.addData('Feedback_block.started', Feedback_block.tStart)
            Feedback_block.maxDuration = 5
            # keep track of which components have finished
            Feedback_blockComponents = Feedback_block.components
            for thisComponent in Feedback_block.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Feedback_block" ---
            # if trial has changed, end Routine now
            if isinstance(block, data.TrialHandler2) and thisBlock.thisN != block.thisTrial.thisN:
                continueRoutine = False
            Feedback_block.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > Feedback_block.maxDuration-frameTolerance:
                    Feedback_block.maxDurationReached = True
                    continueRoutine = False
                
                # *fb_2* updates
                
                # if fb_2 is starting this frame...
                if fb_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fb_2.frameNStart = frameN  # exact frame index
                    fb_2.tStart = t  # local t and not account for scr refresh
                    fb_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fb_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fb_2.started')
                    # update status
                    fb_2.status = STARTED
                    fb_2.setAutoDraw(True)
                
                # if fb_2 is active this frame...
                if fb_2.status == STARTED:
                    # update params
                    pass
                
                # *force_end_of_routine_if_pressed_2* updates
                waitOnFlip = False
                
                # if force_end_of_routine_if_pressed_2 is starting this frame...
                if force_end_of_routine_if_pressed_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    force_end_of_routine_if_pressed_2.frameNStart = frameN  # exact frame index
                    force_end_of_routine_if_pressed_2.tStart = t  # local t and not account for scr refresh
                    force_end_of_routine_if_pressed_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(force_end_of_routine_if_pressed_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'force_end_of_routine_if_pressed_2.started')
                    # update status
                    force_end_of_routine_if_pressed_2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(force_end_of_routine_if_pressed_2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(force_end_of_routine_if_pressed_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if force_end_of_routine_if_pressed_2.status == STARTED and not waitOnFlip:
                    theseKeys = force_end_of_routine_if_pressed_2.getKeys(keyList=['f','l','r','left','right'], ignoreKeys=["escape"], waitRelease=False)
                    _force_end_of_routine_if_pressed_2_allKeys.extend(theseKeys)
                    if len(_force_end_of_routine_if_pressed_2_allKeys):
                        force_end_of_routine_if_pressed_2.keys = _force_end_of_routine_if_pressed_2_allKeys[-1].name  # just the last key pressed
                        force_end_of_routine_if_pressed_2.rt = _force_end_of_routine_if_pressed_2_allKeys[-1].rt
                        force_end_of_routine_if_pressed_2.duration = _force_end_of_routine_if_pressed_2_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *fb_sound* updates
                
                # if fb_sound is starting this frame...
                if fb_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fb_sound.frameNStart = frameN  # exact frame index
                    fb_sound.tStart = t  # local t and not account for scr refresh
                    fb_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('fb_sound.started', tThisFlipGlobal)
                    # update status
                    fb_sound.status = STARTED
                    fb_sound.play(when=win)  # sync with win flip
                
                # if fb_sound is stopping this frame...
                if fb_sound.status == STARTED:
                    if bool(False) or fb_sound.isFinished:
                        # keep track of stop time/frame for later
                        fb_sound.tStop = t  # not accounting for scr refresh
                        fb_sound.tStopRefresh = tThisFlipGlobal  # on global time
                        fb_sound.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fb_sound.stopped')
                        # update status
                        fb_sound.status = FINISHED
                        fb_sound.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[fb_sound]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Feedback_block.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Feedback_block.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Feedback_block" ---
            for thisComponent in Feedback_block.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Feedback_block
            Feedback_block.tStop = globalClock.getTime(format='float')
            Feedback_block.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Feedback_block.stopped', Feedback_block.tStop)
            # check responses
            if force_end_of_routine_if_pressed_2.keys in ['', [], None]:  # No response was made
                force_end_of_routine_if_pressed_2.keys = None
            block.addData('force_end_of_routine_if_pressed_2.keys',force_end_of_routine_if_pressed_2.keys)
            if force_end_of_routine_if_pressed_2.keys != None:  # we had a response
                block.addData('force_end_of_routine_if_pressed_2.rt', force_end_of_routine_if_pressed_2.rt)
                block.addData('force_end_of_routine_if_pressed_2.duration', force_end_of_routine_if_pressed_2.duration)
            fb_sound.pause()  # ensure sound has stopped at end of Routine
            # the Routine "Feedback_block" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed all_phases_data[phase_configs[phase_loop.thisN]['phase_name']]['num_blocks'] repeats of 'block'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        thisExp.nextEntry()
        
    # completed len(phase_configs) repeats of 'phase_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "exp_finished" ---
    # create an object to store info about Routine exp_finished
    exp_finished = data.Routine(
        name='exp_finished',
        components=[text_instr, thank_you],
    )
    exp_finished.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    thank_you.setSound(thank_you_audio, hamming=True)
    thank_you.setVolume(1.0, log=False)
    thank_you.seek(0)
    # Run 'Begin Routine' code from trigger_trial_block_end
    # End of main experiment trial block
    dev.activate_line(bitmask=block_end_code)
    # no need to wait 500ms as this routine lasts 3.0s before experiment ends
    
    # store start times for exp_finished
    exp_finished.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    exp_finished.tStart = globalClock.getTime(format='float')
    exp_finished.status = STARTED
    thisExp.addData('exp_finished.started', exp_finished.tStart)
    exp_finished.maxDuration = None
    # keep track of which components have finished
    exp_finishedComponents = exp_finished.components
    for thisComponent in exp_finished.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "exp_finished" ---
    exp_finished.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instr* updates
        
        # if text_instr is starting this frame...
        if text_instr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instr.frameNStart = frameN  # exact frame index
            text_instr.tStart = t  # local t and not account for scr refresh
            text_instr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instr, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instr.started')
            # update status
            text_instr.status = STARTED
            text_instr.setAutoDraw(True)
        
        # if text_instr is active this frame...
        if text_instr.status == STARTED:
            # update params
            pass
        
        # *thank_you* updates
        
        # if thank_you is starting this frame...
        if thank_you.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thank_you.frameNStart = frameN  # exact frame index
            thank_you.tStart = t  # local t and not account for scr refresh
            thank_you.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('thank_you.started', tThisFlipGlobal)
            # update status
            thank_you.status = STARTED
            thank_you.play(when=win)  # sync with win flip
        
        # if thank_you is stopping this frame...
        if thank_you.status == STARTED:
            if bool(False) or thank_you.isFinished:
                # keep track of stop time/frame for later
                thank_you.tStop = t  # not accounting for scr refresh
                thank_you.tStopRefresh = tThisFlipGlobal  # on global time
                thank_you.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thank_you.stopped')
                # update status
                thank_you.status = FINISHED
                thank_you.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[thank_you]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            exp_finished.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in exp_finished.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "exp_finished" ---
    for thisComponent in exp_finished.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for exp_finished
    exp_finished.tStop = globalClock.getTime(format='float')
    exp_finished.tStopRefresh = tThisFlipGlobal
    thisExp.addData('exp_finished.stopped', exp_finished.tStop)
    thank_you.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "exp_finished" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # Run 'End Experiment' code from eeg
    # Stop EEG recording
    dev.activate_line(bitmask=127)  # trigger 127 will stop EEG
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)

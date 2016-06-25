# Scrumboard

This application allows you to use your physical scrum board as an interface to your digital scrumboard.

It tracks post-it notes on the physical board using a simple webcam and without significant restrictions
on the notes or your way of working.

Add a projector and it turns your physical board into an augmented-reality scrum board, highlighting
overdue tasks for instance.

# How does it work

1. After the camera has been mounted in a fixed position the calibration script is run to select the
   outlines of the board in the frame, the location of the columns (TODO, IN PROGRESS, etc), the size of a
   note and a spot on the board for detecting the board color
   If a projector is installed, it is also calibrated at this time.
2. at the start of a sprint all task notes are put in the TODO column on the scrum board such that
   they do not overlap
3. the application is started for a new sprint and will ask to enter the task IDs for all notes that
   it finds in the image. This creates the association between notes on the physical board and tasks
   in the digital system.
4. from then on the application takes new snapshots of the board every few minutes and checks to see
   if any of the notes have been moved and updates their status accordingly
5. If any new notes are detected, the user is asked to enter a task ID for them

# Current status

The algorithm for recognizing notes on a scrum board and finding them again in later images is currently being implemented.

# TODO

* Implement a connection to digital scrum board software, such as Jira

# Known limitations

* The scrumboard must be white, gray or black, while task notes must have other colors.
* Only square task notes are supported

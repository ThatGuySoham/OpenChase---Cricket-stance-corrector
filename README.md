# Codezilla Kinetic Vision

Pose analysis and form feedback for cricket technique. Runs in a browser tab from one laptop.

## Start it

1. Install Python 3.10, 3.11 or 3.12 from python.org (on Windows, tick "Add Python to PATH").
2. Windows: double-click `run.bat`. Mac or Linux: run `./run.sh`.
3. The first launch installs everything (a few minutes, needs internet). Later launches take seconds.
4. Your browser opens at http://localhost:8501.

## Presenting

- Leave the source on **Simulation**. It needs no camera, no video and no internet, and it shows every feature.
- Press **Start analysis**. Press **Stop** to freeze the screen.
- Switch **Technique** between Batting stance and Bowling action to change what is measured.
- For a volunteer, choose **Webcam**, ask them to stand side-on with their whole body in frame, then Start.
- For a recorded clip, choose **Upload a clip** (mp4, mov or avi, side-on view works best).
- Press F11 in the browser for full screen. Collapse the sidebar with the arrow at its top.
- The **How it works** and **Careers** tabs are written for the career advice part of the session.

## Before the event

- Run it once with internet so the packages and fonts download. The app works offline afterwards, but the fonts fall back to Arial.
- Do a dry run on the projector or screen you will use.
- Test the webcam. Close Zoom, Teams and similar apps first because only one program can use the camera.

## Troubleshooting

| Problem | Fix |
|---|---|
| Sidebar says the pose model is not installed | You are on Python 3.13 or newer, which MediaPipe does not support. Install Python 3.12 and delete the `.venv` folder, then run the launcher again. Simulation still works. |
| Webcam will not open | Close other apps using the camera and allow camera access for Terminal or Python in system privacy settings. |
| No player in view | Step back so the whole body is visible, and use even lighting. |
| Port in use | Run `streamlit run app.py --server.port 8502`. |

## Files

- `app.py` is the whole application.
- `requirements.txt` lists the packages with pinned versions.
- `.streamlit/config.toml` sets the dark theme and turns off usage tracking.
- `run.bat` and `run.sh` pick Python 3.12 (or 3.11, 3.10), build an isolated environment, check that the pose model loads, then launch the app. They never use Python 3.13 or newer.

Send the whole folder as a zip. Do not include the `.venv` folder if it exists.

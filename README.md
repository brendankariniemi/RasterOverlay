# IDS & SCSU Hackathon Feb. 2024 Challenge 2 - Raster Overlay Trimming

## Team: *Team Rocket*
 - Brendan Kariniemi, Rylan Loukusa

## Challenge Description
Currently, the IDS team performs a lot of operations manually that we need to automate to become a true Platform as a Service (PaaS) company for digital twins. One such operation is trimming raster overlay imagery. For example, if a client has a floor plan PDF they would like available as a map overlay, we don’t want the entire pdf to be shown on the map with whitespace all around it, we want to trim the white space and match the floor plan outline. We would like this to be an automated process that happens once a plan is uploaded to our platform.

## Pillar
Compose

## Technologies
 - [PDF](https://www.adobe.com/acrobat/about-adobe-pdf.html)
 - [PNG](https://shorthand.com/the-craft/what-is-a-png-file/index.html)

## Test Data
This challenge uses PDF files for testing.

## Target Outcome
App prototype showcasing conversion of PDF plans to PNG versions where white space surrounding plan elements is made transparent. See `/test-data/examples` folder for some examples of input PDF files and the corresponding output PNG files.

## How to Run
- Open terminal and run the following commands:
- `git clone https://github.com/Immersion-Data-Solutions/TeamRocket.git`
- `python -m venv venv`
- `source venv/bin/activate` for MAC, `venv\Scripts\activate.bat` for Windows
- `pip install -r requirements.txt`
- `flask run`


## Judging Feedback
**What went well:**
-   Completed the hackathon checklist
-   Presentation
    -   All team members spoke
    -   Explained the problem, solution, and areas to improve
        -   The explanation of the two use cases being addressed and the way those use cases were solved was really well done
    -   Live demo showcasing the solution
-   Solution
    -   The end solution was well implemented and could add immediate value to our business
    -   Island highlighting and selection provides a strong user experience for non technical users
    -   Island splitting into separate files supports multi-page / multi-level PDFs really well
    -   Implemented both frontend and backend
    -   Folder selection


## Culture Index Survey
Immersion Data Solutions has engaged with a behavior assessment company, Culture Index. We use it to better understand the intrinsic motivational needs and preferred communication style of our team members; ensuring their current roles fit their strengths and fulfillment needs.

We are pleased to offer you the chance to take the assessment and view your results. The process takes roughly 8 minutes or less. By participating, you will gain insights into your own behavioral tendencies and preferences, which can be invaluable for personal and professional development.

[Please begin assessment here.](https://surveys.cultureindex.com/s/jytaqq125Q/48857) Within the 'job title' field, please enter  **SCSU**  and we will share the results with you.
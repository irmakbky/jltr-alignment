window.alsco = window.alsco || {};

(function (JSZip, alsco) {
    const DEFAULT_MODE = "measures";
    const PALETTE = alsco.PALETTE;

    let CURRENT_PROJECT = null;
    let CURRENT_PROJECT_SERVER_UUID = null;
    let CURRENT_PROJECT_CLIENT_HASH = null;
    let CURRENT_PAGE_INDEX = 0;
    let CURRENT_SELECTED_MEASURE = null;
    let CURRENT_CANVAS_HOVER_POSITION = null;
    let CURRENT_CANVAS_ACTION = null;
    let CURRENT_CANVAS_ACTION_START_POSITION = null;
    let CURRENT_CANVAS_ACTION_START_BBOX = null;

    let CANVAS_EL = null;
    let CANVAS_DEFAULT_HEIGHT = null;
    let ANIMATION_ID = null;

    let TRANSPORT = null;
    let TRANSPORT_SLIDER = null;

    // Graphics

    async function restartAnimationLoop() {
        if (CURRENT_PROJECT === null) return;
        const img = await CURRENT_PROJECT.getPage(
            CURRENT_PAGE_INDEX
        ).getImage();
        function animate() {
            const mode = getCurrentMode();
            const canvasEl = CANVAS_EL;
            const ctx = canvasEl.getContext("2d");

            // Resize
            canvasEl.width = img.width * (canvasEl.height / img.height);

            // Draw page
            ctx.drawImage(img, 0, 0, canvasEl.width, canvasEl.height);

            // Draw mode
            let restart = false;
            if (mode === "measures") {
                CURRENT_PROJECT.drawMeasuresOnCanvasEl(
                    CANVAS_EL,
                    CURRENT_PAGE_INDEX
                );
            } else if (mode === "repeats") {
                CURRENT_PROJECT.drawLogicalOrderOnCanvasEl(
                    CANVAS_EL,
                    CURRENT_PAGE_INDEX
                );
            } else if (mode === "annotations") {
                document.getElementById("keys-input").style.display = "block";
                CURRENT_PROJECT.drawMeasuresOnCanvasEl(
                    CANVAS_EL,
                    CURRENT_PAGE_INDEX
                );
            } else if (mode === "align") {
                // Update transport
                const time = TRANSPORT.getPlayhead();
                if (TRANSPORT.audioBuffer !== null) {
                    TRANSPORT_SLIDER.value = (time / TRANSPORT.duration) * 1000;
                }

                // Update page rendering
                if (CURRENT_PROJECT.alignment !== null) {
                    // Get playhead
                    const playhead =
                        CURRENT_PROJECT.alignment.timeToPlayhead(time);
                    const playheadPageIndex = CURRENT_PROJECT.getPageIndex(
                        playhead.page
                    );

                    // Draw playhead or change page
                    if (playheadPageIndex === CURRENT_PAGE_INDEX) {
                        CURRENT_PROJECT.drawScorePlayheadOnCanvasEl(
                            CANVAS_EL,
                            playhead
                        );
                    } else {
                        restart = true;
                        changePage(playheadPageIndex);
                    }
                }
            }

            // Draw hovered box
            if (CURRENT_CANVAS_HOVER_POSITION !== null) {
                const hoveredMeasures = CURRENT_PROJECT.getPage(
                    CURRENT_PAGE_INDEX
                ).getMeasures(
                    CURRENT_CANVAS_HOVER_POSITION[0],
                    CURRENT_CANVAS_HOVER_POSITION[1]
                );
                if (hoveredMeasures.length > 0) {
                    ctx.strokeStyle = "black";
                    ctx.lineWidth = 2;
                    hoveredMeasures[0].bbox.drawOnCanvas(CANVAS_EL);
                }
            }

            // Draw selected box
            if (CURRENT_SELECTED_MEASURE !== null) {
                ctx.strokeStyle = PALETTE[3];
                ctx.lineWidth = 2;
                CURRENT_SELECTED_MEASURE.bbox.drawOnCanvas(CANVAS_EL);
            }

            // Draw new box
            if (
                CURRENT_CANVAS_ACTION == "crosshair" &&
                CURRENT_CANVAS_ACTION_START_POSITION !== null
            ) {
                ctx.strokeStyle = "black";
                ctx.lineWidth = 2;
                ctx.strokeRect(
                    CURRENT_CANVAS_ACTION_START_POSITION[0] * canvasEl.width,
                    CURRENT_CANVAS_ACTION_START_POSITION[1] * canvasEl.height,
                    (CURRENT_CANVAS_HOVER_POSITION[0] -
                        CURRENT_CANVAS_ACTION_START_POSITION[0]) *
                        canvasEl.width,
                    (CURRENT_CANVAS_HOVER_POSITION[1] -
                        CURRENT_CANVAS_ACTION_START_POSITION[1]) *
                        canvasEl.height
                );
            }

            // Restart or request next frame
            if (restart) {
                ANIMATION_ID = null;
                restartAnimationLoop();
            } else {
                ANIMATION_ID = requestAnimationFrame(animate);
            }
        }
        if (ANIMATION_ID !== null) {
            cancelAnimationFrame(ANIMATION_ID);
        }
        animate();
    }

    // Canvas interaction

    function onCanvasInteract(e) {
        if (CURRENT_PROJECT === null) return;
        const x = e.offsetX / CANVAS_EL.width;
        const y = e.offsetY / CANVAS_EL.height;
        const mode = getCurrentMode();
        const measures = CURRENT_PROJECT.getPage(
            CURRENT_PAGE_INDEX
        ).getMeasures(x, y, true);
        if (e.type === "mousemove") {
            CURRENT_CANVAS_HOVER_POSITION = [x, y];
            if (CURRENT_CANVAS_ACTION_START_POSITION === null) {
                let new_canvas_action = CURRENT_CANVAS_ACTION;
                if (mode === "measures") {
                    if (measures.length > 0) {
                        const selected = measures[0].bbox;
                        // If within 10% of edge, change cursor to pointer
                        if (
                            x <= selected.left + 0.1 * selected.width ||
                            x >= selected.left + 0.9 * selected.width ||
                            y <= selected.top + 0.1 * selected.height ||
                            y >= selected.top + 0.9 * selected.height
                        ) {
                            new_canvas_action = null; //"grabbing";
                        } else {
                            new_canvas_action = null;
                        }
                    } else {
                        new_canvas_action = "crosshair";
                    }
                } else if (mode === "repeats") {
                    new_canvas_action = measures.length > 0 ? "pointer" : null;
                }
                CURRENT_CANVAS_ACTION = new_canvas_action;
                CANVAS_EL.style.cursor =
                    CURRENT_CANVAS_ACTION === null
                        ? "default"
                        : CURRENT_CANVAS_ACTION;
            } else {
                if (CURRENT_CANVAS_ACTION === "grabbing") {
                    CURRENT_CANVAS_ACTION_START_BBOX.left = x;
                    CURRENT_CANVAS_ACTION_START_BBOX.top = y;
                }
            }
        } else if (e.type === "mousedown") {
            if (e.button !== 0) return;
            CURRENT_CANVAS_ACTION_START_POSITION =
                CURRENT_CANVAS_HOVER_POSITION;
            CURRENT_CANVAS_ACTION_START_BBOX =
                measures.length > 0 ? measures[0].bbox : null;
        } else if (e.type === "mouseup") {
            if (e.button !== 0) return;
            if (CURRENT_CANVAS_ACTION === "pointer") {
                if (measures.length > 0) {
                    const selectedMeasure = measures[0];
                    if (CURRENT_SELECTED_MEASURE === null) {
                        CURRENT_SELECTED_MEASURE = selectedMeasure;
                    } else {
                        if (selectedMeasure === CURRENT_SELECTED_MEASURE) {
                            CURRENT_SELECTED_MEASURE = null;
                        } else if (mode === "repeats") {
                            CURRENT_PROJECT.addJump(
                                CURRENT_SELECTED_MEASURE,
                                selectedMeasure
                            );
                            CURRENT_SELECTED_MEASURE = null;
                        }
                    }
                }
            } else if (CURRENT_CANVAS_ACTION === "crosshair") {
                const newBoundingBox = new alsco.types.BoundingBox(
                    Math.min(x, CURRENT_CANVAS_ACTION_START_POSITION[0]),
                    Math.min(y, CURRENT_CANVAS_ACTION_START_POSITION[1]),
                    Math.abs(x - CURRENT_CANVAS_ACTION_START_POSITION[0]),
                    Math.abs(y - CURRENT_CANVAS_ACTION_START_POSITION[1])
                );
                const newMeasure = new alsco.types.Measure(newBoundingBox);
                CURRENT_PROJECT.addMeasure(CURRENT_PAGE_INDEX, newMeasure);
            }
            CURRENT_CANVAS_ACTION = null;
            CURRENT_CANVAS_ACTION_START_POSITION = null;
        } else if (e.type === "mouseout") {
            CURRENT_CANVAS_ACTION = null;
            CURRENT_CANVAS_ACTION_START_POSITION = null;
        } else if (e.type === "contextmenu") {
            if (measures.length > 0) {
                const selectedMeasure = measures[0];
                if (
                    CURRENT_SELECTED_MEASURE !== null &&
                    selectedMeasure === CURRENT_SELECTED_MEASURE
                ) {
                    CURRENT_SELECTED_MEASURE = null;
                }
                if (mode === "measures") {
                    CURRENT_PROJECT.removeMeasure(selectedMeasure);
                } else if (mode === "repeats") {
                    CURRENT_PROJECT.removeJumps(selectedMeasure);
                }
            }
            e.preventDefault();
        }
    }

    // Edit mode

    function getCurrentMode() {
        return document.querySelector("input[name=edit-mode]:checked").value;
    }

    function setCurrentMode(mode) {
        if (!["measures", "repeats", "annotations", "align"].includes(mode)) {
            throw new Error("Invalid mode");
        }
        document.querySelectorAll(".mode-ui").forEach((el) => {
            el.style.display = el.id === `mode-${mode}-ui` ? "block" : "none";
        });
        document.getElementById(`mode-${mode}`).checked = true;
    }

    // Page nav

    function changePage(newPageIndex) {
        if (CURRENT_PROJECT === null) return;
        newPageIndex = Math.max(0, newPageIndex);
        newPageIndex = Math.min(
            CURRENT_PROJECT.getNumPages() - 1,
            newPageIndex
        );
        if (newPageIndex === CURRENT_PAGE_INDEX) return;
        CURRENT_PAGE_INDEX = newPageIndex;
        document.getElementById("nav-current").innerText =
            CURRENT_PAGE_INDEX + 1;
        document.getElementById("nav-total").innerText =
            CURRENT_PROJECT.getNumPages();
    }

    function onChangePage(newPageIndex) {
        changePage(newPageIndex);
        restartAnimationLoop();
    }

    // Canvas zoom

    function onChangeZoomSlider() {
        const slider = document.getElementById("zoom");
        const zoom = slider.value / 1000;
        CANVAS_EL.height = CANVAS_DEFAULT_HEIGHT * zoom;
    }

    // Project state manipulation

    async function maybeSetProject(newProjectPromise, throwAlert = true) {
        function setProject(newProject) {
            CURRENT_PROJECT = newProject;
            CURRENT_PAGE_INDEX = null;
            CURRENT_CANVAS_HOVER_POSITION = null;
            if (newProject.audio !== null)
                TRANSPORT.setAudioBuffer(newProject.audio.audioBuffer);
            setCurrentMode(DEFAULT_MODE);
            onChangePage(0);
        }

        const loadedEl = document.getElementById("project-ui-loaded");
        const spinnerEl = document.getElementById("project-ui-spinner");
        spinnerEl.style.display = "block";
        loadedEl.style.display = "none";
        try {
            const newProject = await newProjectPromise;
            if (!(newProject instanceof alsco.types.Project)) {
                throw new Error("Invalid project");
            }
            if (newProject.pages.length === 0) {
                throw new Error("No page images");
            }
            setProject(newProject);
            loadedEl.style.display = "block";
        } catch (error) {
            console.error("Error:", error);
            if (throwAlert)
                alert("Error initializing project. See console for details.");
        }
        spinnerEl.style.display = "none";
    }

    async function onClickNewProject() {
        let pdfFile;
        try {
            pdfFile = await alsco.helpers.openFilePromise(".pdf");
        } catch (error) {
            return;
        }
        const pdfFileBytes = await pdfFile
            .arrayBuffer()
            .then((buffer) => new Uint8Array(buffer));
        const projectPromise = alsco.api
            .callApiPdfFileToPageImages(pdfFile)
            .then((pageImages) => {
                let pagesBboxesPromises = Promise.all(
                    pageImages.map((pageImage) =>
                        alsco.api.callDetectMeasures(pageImage)
                    )
                );
                return pagesBboxesPromises.then((pagesBboxes) => {
                    const pages = pagesBboxes.map((pageBboxesObjs, index) => {
                        const measures = pageBboxesObjs.map((bboxObj) =>
                            alsco.types.Measure.fromObject({ bbox: bboxObj })
                        );
                        return new alsco.types.Page(
                            measures,
                            pageImages[index]
                        );
                    });
                    return new alsco.types.Project(
                        pages,
                        null,
                        null,
                        null,
                        {},
                        pdfFileBytes,
                        pages
                    );
                });
            });
        maybeSetProject(projectPromise);
    }

    async function onClickLoadProject() {
        let zipFile;
        try {
            zipFile = await alsco.helpers.openFilePromise(".zip");
        } catch (error) {
            return;
        }
        const zip = new JSZip();
        const projectPromise = zip
            .loadAsync(zipFile)
            .then((zip) => alsco.types.Project.fromZip(zip));
        maybeSetProject(projectPromise);
    }

    async function onClickSaveProject() {
        if (CURRENT_PROJECT === null) {
            alert("No project to save");
        } else {
            const zip = CURRENT_PROJECT.toZip();
            const blob = await zip.generateAsync({ type: "blob" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "project.zip";
            a.click();
        }
    }

    async function onClickLoadAudio() {
        if (CURRENT_PROJECT === null) {
            alert("Must load a score before audio.");
        } else {
            let audio;
            try {
                const audioFile = await alsco.helpers.openFilePromise(
                    ".mp3,.ogg,.aiff,.wav,.flac"
                );
                audio = new alsco.types.Audio(
                    await audioFile.arrayBuffer(),
                    audioFile.name
                );
                await audio.initialize();
            } catch (error) {
                alert("Failed to decode audio file.");
                return;
            }
            CURRENT_PROJECT.setAudio(audio);
            TRANSPORT.setAudioBuffer(audio.audioBuffer);
        }
    }

    async function onClickAlign() {
        if (CURRENT_PROJECT === null) {
            alert("No project to align");
        } else {
            // measure indices (mi) are logical, times (ti) are in seconds
            hardConstraints = null;

            // Check for changes and reupload if necessary
            const hashObj = CURRENT_PROJECT.attrsForHash();
            delete hashObj["alignment"];
            console.log(hashObj);
            const hash = alsco.helpers.hashObj(hashObj);
            console.log(hash);
            if (hash !== CURRENT_PROJECT_CLIENT_HASH)
                CURRENT_PROJECT_SERVER_UUID = null;

            // Realign
            maybeSetProject(
                alsco.api
                    .callAlign(
                        CURRENT_PROJECT,
                        CURRENT_PROJECT_SERVER_UUID,
                        hardConstraints
                    )
                    .then(([newUuid, alignment]) => {
                        CURRENT_PROJECT_SERVER_UUID = newUuid;
                        CURRENT_PROJECT_CLIENT_HASH = hash;
                        const newAlignment =
                            new alsco.types.MeasureAwareAlignment(
                                alignment.measure_indices,
                                alignment.times,
                                CURRENT_PROJECT.alignment.logicalOrder,
                                alignment.max_time
                            );
                        CURRENT_PROJECT.alignment = newAlignment;
                        return CURRENT_PROJECT;
                    })
            );
        }
    }

    async function onClickAddKeys() {
        if (CURRENT_PROJECT === null) {
            alert("No project to annotate");
        } else {
            try {
                const measureNumber =
                    document.getElementById("key-measure-num").value;
                const keys = document.getElementById("keys").value;

                if (measureNumber && keys) {
                    const listItem = document.createElement("li");
                    listItem.textContent = `Measure ${measureNumber}: ${keys}`;
                    document.getElementById("keys-list").appendChild(listItem);

                    document.getElementById("key-measure-num").value = "";
                    document.getElementById("keys").value = "";
                } else {
                    alert("Enter both measure number and key signatures.");
                }
            } catch (error) {
                return;
            }
        }
    }

    async function onClickSaveKeys() {
        if (CURRENT_PROJECT === null) {
            alert("No project to annotate");
        } else {
            try {
                const keysList = document.getElementById("keys-list").children;
                const keysData = {};

                for (let item of keysList) {
                    const text = item.textContent;
                    const [measure, keys] = text.split(": ");
                    const measureNumber = measure.split(" ")[1];
                    keysData[measureNumber] = keys;
                }

                const keysInOrder = [];
                for (let page of CURRENT_PROJECT.pages) {
                    const measureBboxes = page.getMeasureBboxes();
                    let lastKey = "C_MAJOR"; // default key, will be set to first measure key if annotated
                    for (let i = 0; i < measureBboxes.length; i++) {
                        let measureNum = (i + 1).toString();
                        if (keysData.hasOwnProperty(measureNum)) {
                            keysInOrder.push(keysData[measureNum]);
                            lastKey = keysData[measureNum];
                        } else {
                            keysInOrder.push(lastKey);
                        }
                    }
                }

                CURRENT_PROJECT.keys = keysInOrder;
            } catch (error) {
                return;
            }
        }
    }

    async function onClickSaveTransposition() {
        if (CURRENT_PROJECT === null) {
            alert("No project to annotate");
        } else {
            const transposition =
                document.getElementById("transposition").value;
            CURRENT_PROJECT.transposition = transposition;
        }
    }

    // Main entry point

    async function onDomReady() {
        // Bind all mouse events for canvas
        CANVAS_EL = document.getElementById("page-canvas");
        CANVAS_EL.addEventListener("click", onCanvasInteract);
        CANVAS_EL.addEventListener("contextmenu", onCanvasInteract);
        CANVAS_EL.addEventListener("mousemove", onCanvasInteract);
        CANVAS_EL.addEventListener("mousedown", onCanvasInteract);
        CANVAS_EL.addEventListener("mouseup", onCanvasInteract);
        CANVAS_DEFAULT_HEIGHT = CANVAS_EL.height;

        // Bind mode radio buttons
        document.querySelectorAll("input[name=edit-mode]").forEach((radio) => {
            radio.addEventListener("change", function () {
                setCurrentMode(getCurrentMode());
                restartAnimationLoop();
            });
        });

        // Bind navigation
        document
            .getElementById("nav-start")
            .addEventListener("click", function () {
                onChangePage(0);
            });
        document
            .getElementById("nav-prev")
            .addEventListener("click", function () {
                onChangePage(CURRENT_PAGE_INDEX - 1);
            });
        document
            .getElementById("nav-next")
            .addEventListener("click", function () {
                onChangePage(CURRENT_PAGE_INDEX + 1);
            });
        document
            .getElementById("nav-end")
            .addEventListener("click", function () {
                onChangePage(1000000);
            });

        // Bind canvas zoom
        document
            .getElementById("zoom")
            .addEventListener("input", onChangeZoomSlider);

        // Bind project state buttons
        document
            .getElementById("new-project")
            .addEventListener("click", onClickNewProject);
        document
            .getElementById("load-project")
            .addEventListener("click", onClickLoadProject);
        document
            .getElementById("save-project")
            .addEventListener("click", onClickSaveProject);

        // Bind transport
        TRANSPORT = new alsco.types.Transport(new AudioContext());
        TRANSPORT_SLIDER = document.getElementById("transport");
        TRANSPORT_SLIDER.addEventListener("input", function () {
            if (TRANSPORT.audioBuffer !== null) {
                TRANSPORT.setPlayhead(
                    (TRANSPORT_SLIDER.value / 1000) * TRANSPORT.duration
                );
            }
        });
        document
            .getElementById("transport-play")
            .addEventListener("click", () => {
                TRANSPORT.start();
            });
        document
            .getElementById("transport-pause")
            .addEventListener("click", () => {
                TRANSPORT.pause();
            });
        document
            .getElementById("transport-stop")
            .addEventListener("click", () => {
                TRANSPORT.restart();
            });

        // Bind align
        document
            .getElementById("load-audio")
            .addEventListener("click", onClickLoadAudio);
        document
            .getElementById("align")
            .addEventListener("click", onClickAlign);

        // Bind annotations
        document
            .getElementById("add-keys")
            .addEventListener("click", onClickAddKeys);

        document
            .getElementById("save-keys")
            .addEventListener("click", onClickSaveKeys);

        document
            .getElementById("save-transposition")
            .addEventListener("click", onClickSaveTransposition);

        if (alsco.DEBUG) {
            maybeSetProject(
                alsco.api
                    .fetchFile("/mesa13/maple_leaf_rag.zip")
                    .then((zipFile) => {
                        const zip = new JSZip();
                        return zip
                            .loadAsync(zipFile)
                            .then((zip) => alsco.types.Project.fromZip(zip));
                    }),
                false
            );
        }
    }

    document.addEventListener("DOMContentLoaded", onDomReady);
})(window.JSZip, window.alsco);

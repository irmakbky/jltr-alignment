window.alsco = window.alsco || {};

(function (JSZip, alsco) {
    const DISCRETIZATION = 1_000_000;
    const PALETTE = alsco.PALETTE;

    function discretize(x) {
        return Math.round(x * DISCRETIZATION);
    }

    class Transport {
        constructor(audioContext, audioBuffer) {
            this.audioContext = audioContext;
            this.active = false;
            this.playhead = 0.0;
            this.startTime = null;
            this.duration = null;
            this.source = null;
            this.audioBuffer = null;
            this.setAudioBuffer(audioBuffer);
        }

        setAudioBuffer(audioBuffer) {
            if (!(audioBuffer instanceof AudioBuffer)) return;
            this.restart();
            this.audioBuffer = audioBuffer;
            this.duration = audioBuffer.duration;
        }

        getPlayhead() {
            let playhead = this.playhead;
            if (this.active)
                playhead += this.audioContext.currentTime - this.startTime;
            return playhead;
        }

        setPlayhead(playhead) {
            this.pause();
            this.playhead = playhead;
        }

        start() {
            if (this.active) return;
            if (this.audioContext.state !== "running") {
                this.audioContext.resume();
            }
            this.source = this.audioContext.createBufferSource();
            if (this.audioBuffer !== null)
                this.source.buffer = this.audioBuffer;
            this.source.connect(this.audioContext.destination);
            this.active = true;
            this.source.start(0, this.playhead);
            this.source.setB;
            this.startTime = this.audioContext.currentTime;
            console.log(`Start: ${this.playhead}`);
        }

        pause() {
            if (!this.active) return;
            this.active = false;
            this.source.disconnect();
            this.source.stop(0);
            this.source = null;
            this.playhead += this.audioContext.currentTime - this.startTime;
            this.startTime = null;
            console.log(`Pause: ${this.playhead}`);
        }

        restart() {
            this.pause();
            this.active = false;
            this.playhead = 0.0;
            this.startTime = null;
        }
    }

    class Hashable {
        attrsForHash() {
            throw new Error("Not implemented");
        }

        attrsHashed() {
            const attrsObj = this.attrsForHash();
            if (typeof attrsObj === "object") {
                const hashedAttrs = {};
                for (const [key, value] of Object.entries(attrsObj)) {
                    if (value === undefined || value === null) continue;
                    hashedAttrs[key] = alsco.helpers.hashObj(value);
                }
                return hashedAttrs;
            } else {
                return alsco.helpers.hashObj(attrsObj);
            }
        }

        checksum() {
            return alsco.helpers.hashObj(this.attrsForHash());
        }
    }

    class Audio extends Hashable {
        constructor(arrayBuffer, fileName) {
            super();
            if (!(arrayBuffer instanceof ArrayBuffer)) {
                throw new Error("Invalid arrayBuffer type");
            }
            this.arrayBuffer = arrayBuffer;
            this.rawBytes = new Uint8Array(this.arrayBuffer);
            this.ext = fileName.split(".").pop();
            this.audioBuffer = null;
        }

        async initialize(audioContext) {
            audioContext = audioContext ? audioContext : new AudioContext();
            const arrayBufferCopy = this.arrayBuffer.slice(0);
            this.audioBuffer = await audioContext.decodeAudioData(
                arrayBufferCopy
            );
        }

        attrsForHash() {
            return alsco.helpers.sha256(this.rawBytes);
        }
    }

    class BoundingBox extends Hashable {
        constructor(left, top, width, height, score) {
            super();
            if (left === undefined || left < 0 || left > 1) {
                throw new Error("Invalid left value");
            }
            if (top === undefined || top < 0 || top > 1) {
                throw new Error("Invalid top value");
            }
            if (width === undefined || width < 0 || width > 1) {
                throw new Error("Invalid width value");
            }
            if (height === undefined || height < 0 || height > 1) {
                throw new Error("Invalid height value");
            }
            if (left + width > 1) {
                throw new Error("Invalid left/width value");
            }
            if (top + height > 1) {
                throw new Error("Invalid top/height value");
            }
            if (score !== undefined && (score < 0 || score > 1)) {
                throw new Error("Invalid score value");
            }
            this.left = left;
            this.top = top;
            this.width = width;
            this.height = height;
            this.score = score === undefined ? null : score;
        }

        getUniqueKey() {
            const o = this.attrsForHash();
            return `${o.left}-${o.top}-${o.width}-${o.height}`;
        }

        getCenter() {
            return {
                x: this.left + this.width / 2,
                y: this.top + this.height / 2,
            };
        }

        drawOnCanvas(canvasEl) {
            const ctx = canvasEl.getContext("2d");
            const canvasWidth = canvasEl.width;
            const canvasHeight = canvasEl.height;
            ctx.strokeRect(
                this.left * canvasWidth,
                this.top * canvasHeight,
                this.width * canvasWidth,
                this.height * canvasHeight
            );
        }

        attrsForHash() {
            return {
                left: discretize(this.left),
                top: discretize(this.top),
                width: discretize(this.width),
                height: discretize(this.height),
            };
        }

        asObject() {
            return this;
        }

        static fromObject(obj) {
            return new BoundingBox(
                obj.left,
                obj.top,
                obj.width,
                obj.height,
                obj.score
            );
        }
    }

    class Measure extends Hashable {
        constructor(bbox, keySignature, timeSignature, clefs) {
            super();
            if (!(bbox instanceof BoundingBox)) {
                throw new Error("Invalid bbox type");
            }
            this.bbox = bbox;
            this.keySignature =
                keySignature === undefined ? null : keySignature;
            this.timeSignature =
                timeSignature === undefined ? null : timeSignature;
            this.clefs = clefs === undefined ? null : clefs;
        }

        attrsForHash() {
            return {
                bbox: this.bbox.attrsForHash(),
                key_signature: this.keySignature,
                time_signature: this.timeSignature,
                clefs: this.clefs,
            };
        }

        asObject() {
            return {
                bbox: this.bbox.asObject(),
                key_signature: this.keySignature,
                time_signature: this.timeSignature,
                clefs: this.clefs,
            };
        }

        static fromObject(obj) {
            return new Measure(
                BoundingBox.fromObject(obj.bbox),
                obj.key_signature,
                obj.time_signature,
                obj.clefs
            );
        }
    }

    class Page extends Hashable {
        constructor(measures, imageBytes, numStaves) {
            super();
            if (
                !Array.isArray(measures) ||
                !measures.every((m) => m instanceof Measure)
            ) {
                throw new Error("Invalid measures type");
            }
            if (!(imageBytes instanceof Uint8Array)) {
                throw new Error("Invalid imageBytes type");
            }
            this.measures = measures;
            this.imageBytes = imageBytes;
            this.numStaves = numStaves === undefined ? null : numStaves;
            this._imagePromise = null;
        }

        getNumMeasures() {
            return this.measures.length;
        }

        async getImage() {
            if (this._imagePromise === null) {
                const img = new Image();
                img.src =
                    "data:image/png;base64," +
                    alsco.helpers.bytesToBase64(this.imageBytes);
                this._imagePromise = new Promise((resolve, reject) => {
                    img.onload = function () {
                        resolve(img);
                    };
                    img.onerror = function () {
                        reject(new Error("Failed to load page image"));
                    };
                });
            }
            return await this._imagePromise;
        }

        getMeasures(x, y, sortByDistance = false) {
            let result = this.measures;
            if (x !== undefined && y !== undefined) {
                result = result.filter((m) => {
                    return (
                        x >= m.bbox.left &&
                        x <= m.bbox.left + m.bbox.width &&
                        y >= m.bbox.top &&
                        y <= m.bbox.top + m.bbox.height
                    );
                });
                if (sortByDistance) {
                    result.sort((a, b) => {
                        const aCenter = a.bbox.getCenter();
                        const bCenter = b.bbox.getCenter();
                        const aDist = Math.sqrt(
                            Math.pow(aCenter.x - x, 2) +
                                Math.pow(aCenter.y - y, 2)
                        );
                        const bDist = Math.sqrt(
                            Math.pow(bCenter.x - x, 2) +
                                Math.pow(bCenter.y - y, 2)
                        );
                        return aDist - bDist;
                    });
                }
            }
            return result;
        }

        getMeasuresInGraphicalOrder() {
            // Call this.getLines() and flatten the result
            return this.getMeasuresInGraphicalOrderByLine().reduce(
                (acc, line) => acc.concat(line),
                []
            );
        }

        getMeasuresInGraphicalOrderByLine() {
            let measures = this.measures;
            measures.sort((a, b) => {
                return a.bbox.top - b.bbox.top;
            });
            let lines = [];
            while (measures.length > 0) {
                const linePrototype = measures[0].bbox;
                const halfMeasureHeight = linePrototype.height / 2;
                const line = measures.filter((m) => {
                    return (
                        Math.abs(m.bbox.top - linePrototype.top) <
                        halfMeasureHeight
                    );
                });
                line.sort((a, b) => {
                    return a.bbox.left - b.bbox.left;
                });
                lines.push(line);
                measures = measures.filter((m) => {
                    return (
                        Math.abs(m.bbox.top - linePrototype.top) >=
                        halfMeasureHeight
                    );
                });
            }
            return lines;
        }

        addMeasure(measure) {
            this.measures.push(measure);
        }

        removeMeasure(measure) {
            // by reference
            let index = this.measures.indexOf(measure);
            if (index === -1) {
                throw new Error("Invalid bounding box");
            }
            this.measures.splice(index, 1);
        }

        attrsForHash() {
            const result = {
                measures: this.getMeasuresInGraphicalOrder().map((measure) =>
                    measure.attrsForHash()
                ),
                num_staves: this.numStaves,
            };
            result.image = null;
            if (this.imageBytes !== null) {
                result.image = alsco.helpers.sha256(this.imageBytes);
            }
            return result;
        }

        asObject() {
            return {
                measures: this.getMeasuresInGraphicalOrder().map((measure) =>
                    measure.asObject()
                ),
                num_staves: this.numStaves,
            };
        }

        static fromObject(obj, imageBytes) {
            return new Page(
                obj.measures.map(Measure.fromObject),
                imageBytes,
                obj.num_staves
            );
        }
    }

    class Monotonic1DLinterp {
        constructor(xCoords, yCoords) {
            if (!Array.isArray(xCoords) || !Array.isArray(yCoords)) {
                throw new Error("Invalid coords type");
            }
            if (xCoords.length !== yCoords.length) {
                throw new Error("Coords should have same length");
            }
            if (xCoords.length < 2) {
                throw new Error("Coords should have at least 2 elements");
            }
            for (let i = 1; i < xCoords.length; ++i) {
                if (xCoords[i] < xCoords[i - 1]) {
                    throw new Error("xCoords should be non-decreasing");
                }
            }
            for (let i = 1; i < yCoords.length; ++i) {
                if (yCoords[i] < yCoords[i - 1]) {
                    throw new Error("yCoords should be non-decreasing");
                }
            }
            this.xCoords = xCoords;
            this.yCoords = yCoords;
        }

        interp(x) {
            if (
                x < this.xCoords[0] ||
                x > this.xCoords[this.xCoords.length - 1]
            )
                return null;
            let i = 0;
            while (this.xCoords[i + 1] < x) ++i;
            return (
                this.yCoords[i] +
                ((this.yCoords[i + 1] - this.yCoords[i]) /
                    (this.xCoords[i + 1] - this.xCoords[i])) *
                    (x - this.xCoords[i])
            );
        }
    }

    function measureOrderAttrsForHash(measureOrder) {
        const uniquePage = [];
        for (const [page, _] of measureOrder) {
            if (!uniquePage.includes(page)) {
                uniquePage.push(page);
            }
        }
        return measureOrder.map(([page, measure]) => [
            uniquePage.indexOf(page),
            measure.attrsForHash(),
        ]);
    }

    class MeasureAwareAlignment extends Hashable {
        constructor(measureIndices, times, logicalOrder, maxTime) {
            super();
            this.measureIndices = measureIndices;
            this.times = times;
            this.logicalOrder = logicalOrder;
            this._timeToMeasure = new Monotonic1DLinterp(times, measureIndices);
            this.maxTime = maxTime;
        }

        timeToMeasureIndex(time) {
            return this._timeToMeasure.interp(time);
        }

        timeToPlayhead(time) {
            const measure = Math.max(this.timeToMeasureIndex(time), 0.0);
            const measureIndex = Math.floor(measure);
            const measureFraction = measure - measureIndex;
            const [pageObj, measureObj] = this.logicalOrder[measureIndex];
            const bbox = measureObj.bbox;
            return {
                page: pageObj,
                x: bbox.left + measureFraction * bbox.width,
                y: bbox.top,
                height: bbox.height,
            };
        }

        attrsForHash() {
            return {
                measure_indices: this.measureIndices.map(discretize),
                times: this.times.map(discretize),
                logical_order: measureOrderAttrsForHash(this.logicalOrder),
                max_time: discretize(this.maxTime),
            };
        }
    }

    class Project extends Hashable {
        //                       Details use on frontend (FE) or backend (BE)
        constructor( //         | Purpose                        | FE | BE |
            pages, //           | Page objects                   | 1  | 1  |
            audio, //           | Optional, performance audio    | 1  | 1  |
            alignment, //       | Optional, alignment            | 1  | 1  |
            transposition, //   | Optional, transposition        | 1  | 1  |
            jumps, //           | Jumps between measures         | 1  | -  |
            scorePdfBytes, //   | Score as PDF                   | -  | 1  |
            originalPages //    | Unedited measures (posterity)  | -  | -  |
        ) {
            super();
            function checkPagesTypes(pages) {
                if (
                    !(
                        Array.isArray(pages) &&
                        pages.every((page) => page instanceof Page)
                    )
                ) {
                    throw new Error("Invalid pages type");
                }
                return pages;
            }
            // Check pages
            checkPagesTypes(pages);

            // Check audio
            if (
                !(
                    audio === null ||
                    audio === undefined ||
                    audio instanceof Audio
                )
            ) {
                throw new Error("Invalid audio type");
            }

            // Check alignment
            if (
                !(
                    alignment === null ||
                    alignment === undefined ||
                    alignment instanceof MeasureAwareAlignment
                )
            ) {
                throw new Error("Invalid alignment type");
            }

            // Check scorePdfBytes
            if (!(scorePdfBytes instanceof Uint8Array)) {
                throw new Error("Invalid scorePdfBytes type");
            }

            // Check originalPages
            if (originalPages !== undefined) {
                checkPagesTypes(originalPages);
            }

            this.pages = pages;
            this.audio = audio === undefined ? null : audio;
            this.alignment = alignment === undefined ? null : alignment;
            this.transposition =
                transposition === undefined ? null : transposition;
            this.jumps = jumps;
            this.scorePdfBytes = scorePdfBytes;
            this.originalPages =
                originalPages === undefined ? null : originalPages;
        }

        // Protected helpers

        _ensureValidPageIndex(pageIndex) {
            if (pageIndex < 0 || pageIndex >= this.getNumPages()) {
                throw new Error("Invalid pageIndex");
            }
        }

        _createUniqueMeasureKey(measure) {
            const pageIndex = this.getPageIndex(measure);
            return `${pageIndex},${measure.bbox.getUniqueKey()}`;
        }

        _getMeasureIndexInGraphicalOrder(uniqueKey) {
            let [page, bboxKey] = uniqueKey.split(",");
            page = parseInt(page);
            const measures = this.getPage(page).getMeasuresInGraphicalOrder();
            for (let i = 0; i < measures.length; ++i) {
                if (measures[i].bbox.getUniqueKey() === bboxKey) {
                    return [page, i];
                }
            }
            throw new Error("Invalid bbox");
        }

        // Public getters

        getNumPages() {
            return this.pages.length;
        }

        getPage(pageIndex) {
            this._ensureValidPageIndex(pageIndex);
            return this.pages[pageIndex];
        }

        getPageIndex(obj) {
            if (obj instanceof Page) {
                return this.pages.indexOf(obj);
            } else if (obj instanceof Measure) {
                for (let i = 0; i < this.pages.length; ++i) {
                    const measures = this.pages[i].measures;
                    for (let j = 0; j < measures.length; ++j) {
                        if (measures[j] === obj) {
                            return i;
                        }
                    }
                }
                throw new Error("Measure object doesn't belong to any page");
            } else {
                throw new Error("Invalid object type");
            }
        }

        getNumMeasuresBeforePage(pageIndex) {
            this._ensureValidPageIndex(pageIndex);
            let result = 0;
            for (let i = 0; i < pageIndex; ++i) {
                result += this.pages[i].getNumMeasures();
            }
            return result;
        }

        getPathLegacy() {
            const path = [];
            let page = 0;
            let measure = 0;
            let jump = false;
            const keyToCount = {};
            while (page < this.getNumPages()) {
                const pageMeasures =
                    this.getPage(page).getMeasuresInGraphicalOrder();

                // Iterate through measures
                while (true) {
                    if (measure >= pageMeasures.length) {
                        page = page + 1;
                        measure = 0;
                        break;
                    }

                    // Push measure to path
                    const measureObj = pageMeasures[measure];
                    const bbox = measureObj.bbox;
                    path.push([page, measure, bbox, jump]);

                    // Handle jumps
                    const measureKey = this._createUniqueMeasureKey(measureObj);
                    if (keyToCount[measureKey] === undefined)
                        keyToCount[measureKey] = 0;
                    const seenCount = keyToCount[measureKey];
                    keyToCount[measureKey] += 1;

                    // Handle jumps
                    const jumps = this.jumps[measureKey];
                    if (jumps === undefined) {
                        jump = false;
                        ++measure;
                    } else {
                        jump = true;
                        if (seenCount < jumps.length) {
                            [page, measure] =
                                this._getMeasureIndexInGraphicalOrder(
                                    jumps[seenCount]
                                );
                            break;
                        } else {
                            // Break outer loop (unclear where to go from here)
                            page = this.getNumPages();
                            break;
                        }
                    }
                }
            }

            return path;
        }

        getLogicalOrder() {
            return this.getPathLegacy().map(([page, measure, _, __]) => [
                this.getPage(page),
                this.getPage(page).getMeasuresInGraphicalOrder()[measure],
            ]);
        }

        // Manipulate state

        addJump(srcMeasureObj, destMeasureObj) {
            const srcKey = this._createUniqueMeasureKey(srcMeasureObj);
            const destKey = this._createUniqueMeasureKey(destMeasureObj);
            if (this.jumps[srcKey] === undefined) {
                this.jumps[srcKey] = [];
            }
            this.jumps[srcKey].push(destKey);
        }

        removeJumps(measureObj) {
            const pageIndex = this.getPageIndex(measureObj);
            const key = this._createUniqueMeasureKey(measureObj);
            delete this.jumps[key];
            const deleteKeys = [];
            for (const srcKey in this.jumps) {
                const filtered = this.jumps[srcKey].filter(
                    (destKey) => destKey !== key
                );
                if (filtered.length === 0) {
                    deleteKeys.push(srcKey);
                } else {
                    this.jumps[srcKey] = filtered;
                }
            }
            deleteKeys.forEach((key) => {
                delete this.jumps[key];
            });
        }

        addMeasure(pageIndex, measureObj) {
            this.getPage(pageIndex).addMeasure(measureObj);
        }

        removeMeasure(measureObj) {
            this.removeJumps(measureObj);
            this.getPage(this.getPageIndex(measureObj)).removeMeasure(
                measureObj
            );
        }

        setAudio(audio) {
            this.audio = audio;
        }

        // Graphics

        drawMeasuresOnCanvasEl(canvasEl, pageIndex) {
            this._ensureValidPageIndex(pageIndex);
            const page = this.pages[pageIndex];
            const ctx = canvasEl.getContext("2d");
            const canvasWidth = canvasEl.width;
            const canvasHeight = canvasEl.height;
            let measureIndex = this.getNumMeasuresBeforePage(pageIndex);
            page.getMeasuresInGraphicalOrderByLine().forEach(
                (line, lineIndex) => {
                    line.forEach((measure, index) => {
                        const box = measure.bbox;
                        ctx.strokeStyle =
                            PALETTE[(lineIndex % 2) * 2 + (index % 2)];
                        ctx.strokeRect(
                            box.left * canvasWidth,
                            box.top * canvasHeight,
                            box.width * canvasWidth,
                            box.height * canvasHeight
                        );

                        // Write pageIndex/measureIndex on top left corner
                        // Draw white box to make text readable
                        ctx.fillStyle = "white";
                        ctx.fillRect(
                            box.left * canvasWidth,
                            box.top * canvasHeight,
                            20,
                            12
                        );
                        ctx.fillStyle = "black";
                        ctx.font = "12px Arial";
                        ctx.fillText(
                            `m${measureIndex + 1}`,
                            box.left * canvasWidth,
                            box.top * canvasHeight + 12
                        );
                        ++measureIndex;
                    });
                }
            );
        }

        drawLogicalOrderOnCanvasEl(canvasEl, pageIndex) {
            this._ensureValidPageIndex(pageIndex);
            const ctx = canvasEl.getContext("2d");
            const canvasWidth = canvasEl.width;
            const canvasHeight = canvasEl.height;
            const path = this.getPathLegacy();

            const shapes = {};
            const bboxes = {};
            const lines = [];

            function getUniqueStr(i) {
                // 0 -> a, 1 -> b, ..., 26 -> aa, 27 -> ab, ...
                let result = "";
                while (i >= 0) {
                    // upper case
                    result = String.fromCharCode(65 + (i % 26)) + result;
                    i = Math.floor(i / 26) - 1;
                }
                return result;
            }

            // Iterate through path deciding what to draw
            let jumpInstance = 0;
            for (let i = 0; i < path.length; ++i) {
                const [currPage, currMeasure, currBbox, currJump] = path[i];
                if (currPage === pageIndex) {
                    bboxes[currMeasure] = currBbox;
                    if (shapes[currMeasure] === undefined)
                        shapes[currMeasure] = [];
                }

                // Determine shape
                let shape = currJump ? "square" : "circle";
                let color = currJump ? PALETTE[0] : PALETTE[1];
                let text = currJump
                    ? [`←${getUniqueStr(jumpInstance - 1)}`]
                    : [];
                if (i + 1 < path.length) {
                    const [nextPage, nextMeasure, nextBbox, nextJump] =
                        path[i + 1];
                    if (nextJump) {
                        shape = "square";
                        color = PALETTE[0];
                        text.push(`→${getUniqueStr(jumpInstance++)}`);
                    } else if (i === 0) {
                        shape = "square";
                        text.push("<S>");
                    } else if (currPage !== nextPage) {
                        shape = "square";
                        text.push(`→P${nextPage + 1}`);
                    }
                } else {
                    shape = "square";
                    text.push("<E>");
                }

                if (currPage === pageIndex) {
                    // Display shape
                    shapes[currMeasure].push([shape, color, text]);

                    // Display line
                    if (i > 0) {
                        const [prevPage, prevMeasure, prevBbox, prevJump] =
                            path[i - 1];
                        if (
                            prevPage == currPage &&
                            currMeasure === prevMeasure + 1 &&
                            !currJump
                        ) {
                            lines.push([
                                prevMeasure,
                                shapes[prevMeasure].length - 1,
                                currMeasure,
                                shapes[currMeasure].length - 1,
                                currJump ? "red" : "black",
                            ]);
                        }
                    }
                }
            }

            function getShapeOffset(bbox, num, i) {
                // Bbox is left, top, width, height
                // Pack num shapes into it at center and equally distributed across the height
                let result = bbox.getCenter();
                if (num > 1) {
                    result.y = bbox.top + (bbox.height * (i + 1)) / (num + 1);
                }
                return result;
            }

            function drawShapes(measureAndShape) {
                const [measure, s] = measureAndShape;
                if (s.length === 0) return;
                const bbox = bboxes[measure];
                for (let i = 0; i < s.length; ++i) {
                    const a = getShapeOffset(bbox, s.length, i);
                    const shapeType = s[i][0];
                    ctx.strokeStyle = s[i][1];
                    const text = s[i][2];
                    ctx.fillStyle = "white";
                    if (shapeType === "circle") {
                        ctx.beginPath();
                        ctx.arc(
                            a.x * canvasWidth,
                            a.y * canvasHeight,
                            16,
                            0,
                            2 * Math.PI
                        );
                        ctx.fill();
                        ctx.stroke();
                    } else {
                        ctx.fillRect(
                            a.x * canvasWidth - 16,
                            a.y * canvasHeight - 16,
                            30,
                            30
                        );
                        ctx.strokeRect(
                            a.x * canvasWidth - 16,
                            a.y * canvasHeight - 16,
                            30,
                            30
                        );
                    }
                    ctx.fillStyle = "black";
                    for (let i = 0; i < text.length; ++i) {
                        ctx.fillText(
                            text[i],
                            a.x * canvasWidth - 14,
                            a.y * canvasHeight - 2 + i * 12
                        );
                    }
                }
            }

            function drawLine(l) {
                const [aMeasure, aIndex, bMeasure, bIndex, color] = l;
                const a = getShapeOffset(
                    bboxes[aMeasure],
                    shapes[aMeasure].length,
                    aIndex
                );
                const b = getShapeOffset(
                    bboxes[bMeasure],
                    shapes[bMeasure].length,
                    bIndex
                );
                ctx.strokeStyle = color;
                ctx.beginPath();
                ctx.moveTo(a.x * canvasWidth, a.y * canvasHeight);
                ctx.lineTo(b.x * canvasWidth, b.y * canvasHeight);
                ctx.stroke();
            }

            // Set fixed styles
            ctx.lineWidth = 2;
            ctx.font = "12px Arial";
            lines.forEach(drawLine);
            // call iterate with (key, value)
            Object.entries(shapes).forEach(drawShapes);
        }

        drawScorePlayheadOnCanvasEl(canvasEl, playhead) {
            if (this.alignment === null) {
                throw new Error("No alignment available");
            }
            const ctx = canvasEl.getContext("2d");
            const canvasWidth = canvasEl.width;
            const canvasHeight = canvasEl.height;
            ctx.strokeStyle = "red";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(playhead.x * canvasWidth, playhead.y * canvasHeight);
            ctx.lineTo(
                playhead.x * canvasWidth,
                (playhead.y + playhead.height) * canvasHeight
            );
            ctx.stroke();
        }

        // Hash

        attrsForHash() {
            const result = {
                score_pages: this.pages.map((page) => {
                    const pageHash = page.attrsForHash();
                    delete pageHash.image;
                    return pageHash;
                }),
                score_pdf: alsco.helpers.sha256(this.scorePdfBytes),
                transposition: this.transposition,
            };
            // NOTE: Hacky, but works for now
            const isAlignedScore = this.alignment !== null;
            if (isAlignedScore) {
                result.alignment = this.alignment.attrsForHash();
                result.performance = this.audio.attrsForHash();
            } else {
                result.logical_order = measureOrderAttrsForHash(
                    this.getLogicalOrder()
                );
            }
            return result;
        }

        // Zip I/O

        toZip() {
            const zip = new JSZip();

            // Version
            zip.file("version.txt", alsco.VERSION);

            // Checksum
            zip.file("checksum.txt", this.checksum());

            // Pages
            zip.file(
                "pages.json",
                JSON.stringify(
                    this.pages.map((page) => page.asObject()),
                    null,
                    2
                )
            );

            // Write logical order
            let logicalOrder;
            if (this.alignment === null) {
                logicalOrder = this.getLogicalOrder();
            } else {
                logicalOrder = this.alignment.logicalOrder;
            }
            const logicalOrderAsCoordinates = [];
            logicalOrder.forEach(([page, measure]) => {
                const pageIndex = this.getPageIndex(page);
                const measureIndex = page
                    .getMeasuresInGraphicalOrder()
                    .indexOf(measure);
                logicalOrderAsCoordinates.push([pageIndex, measureIndex]);
            });
            zip.file(
                "logical_order.json",
                JSON.stringify(logicalOrderAsCoordinates)
            );

            // Score PDF
            zip.file("score.pdf", this.scorePdfBytes);

            // Transposition
            if (this.transposition !== null) {
                zip.file("transposition.txt", this.transposition.toString());
            }

            // (Optional) Audio
            if (this.audio !== null) {
                zip.file(`performance.${this.audio.ext}`, this.audio.rawBytes);
            }

            // (Optional) Alignment
            if (this.alignment !== null) {
                zip.file(
                    "alignment.json",
                    JSON.stringify({
                        measure_indices: this.alignment.measureIndices,
                        times: this.alignment.times,
                        max_time: this.alignment.maxTime,
                    })
                );
            }

            // Page images
            this.pages.forEach((page, index) => {
                zip.file("frontend/page-" + index + ".png", page.imageBytes);
            });

            // (Optional) Original pages
            if (this.originalPages !== null) {
                zip.file(
                    "frontend/original_pages.json",
                    JSON.stringify(
                        this.originalPages.map((page) => page.asObject()),
                        null,
                        2
                    )
                );
            }

            // Jumps
            zip.file(
                "frontend/jumps.json",
                JSON.stringify(this.jumps, null, 2)
            );

            return zip;
        }

        static async fromZip(zip, checkConsistency = false) {
            // Version
            const version = await zip.file("version.txt").async("string");
            if (version !== alsco.VERSION) {
                throw new Error("Invalid project version");
            }

            // (Guaranteed for now, frontend/page-*.png) Page images
            const pageImagesBytes = [];
            while (true) {
                const pageImage = zip.file(
                    "frontend/page-" + pageImagesBytes.length + ".png"
                );
                if (pageImage === null) break;
                pageImagesBytes.push(await pageImage.async("uint8array"));
            }

            // (Guaranteed; pages.json) Pages
            const pages = await zip
                .file("pages.json")
                .async("string")
                .then((jsonString) => {
                    const pageObjects = JSON.parse(jsonString);
                    if (pageObjects.length !== pageImagesBytes.length) {
                        throw new Error("Inconsistent number of pages");
                    }
                    return pageObjects.map((pageObj, index) =>
                        Page.fromObject(pageObj, pageImagesBytes[index])
                    );
                });

            // (Guaranteed; logical_order.json) Logical order
            let logicalOrder = await zip
                .file("logical_order.json")
                .async("string")
                .then((jsonString) => JSON.parse(jsonString));
            logicalOrder = logicalOrder.map((pageAndMeasureIndex) => {
                const [pageIndex, measureIndex] = pageAndMeasureIndex;
                const page = pages[pageIndex];
                return [page, page.measures[measureIndex]];
            });

            // (Optional; performance.*) Audio
            let audio;
            const audioFile = Object.keys(zip.files).find((fileName) =>
                fileName.startsWith("performance")
            );
            if (audioFile) {
                audio = new Audio(
                    await zip.file(audioFile).async("arraybuffer"),
                    audioFile
                );
                await audio.initialize();
            }

            // (Optional, alignment.json) Alignment
            let alignment;
            if ("alignment.json" in zip.files) {
                const alignmentDict = await zip
                    .file("alignment.json")
                    .async("string")
                    .then((jsonString) => JSON.parse(jsonString));
                alignment = new MeasureAwareAlignment(
                    alignmentDict.measure_indices,
                    alignmentDict.times,
                    logicalOrder,
                    alignmentDict.max_time
                );
            }

            // (Optional, transposition.txt) Transposition
            let transposition;
            if ("transposition.txt" in zip.files) {
                transposition = await zip
                    .file("transposition.txt")
                    .async("string")
                    .then((data) => parseInt(data, 10));
            }

            // (Optional, frontend/jumps.json) Jumps
            let jumps = {};
            if ("frontend/jumps.json" in zip.files) {
                jumps = await zip
                    .file("frontend/jumps.json")
                    .async("string")
                    .then((jsonString) => JSON.parse(jsonString));
            }

            // (Optional, score.pdf) Score PDF
            const pdfBytes = await zip.file("score.pdf").async("uint8array");

            // (Optional, original_pages.json) Original pages
            let originalPages;
            if ("original_pages.json" in zip.files) {
                originalPages = await zip
                    .file("original_pages.json")
                    .async("string")
                    .then((jsonString) =>
                        JSON.parse(jsonString).map((page) =>
                            Page.fromObject(page)
                        )
                    );
            }

            const result = new Project(
                pages,
                audio,
                alignment,
                transposition,
                jumps,
                pdfBytes,
                originalPages
            );

            // Checksum
            const checksum = await zip.file("checksum.txt").async("string");
            if (checksum !== result.checksum()) {
                if (checkConsistency) {
                    throw new Error("Checksum mismatch");
                } else {
                    console.warn("Checksum mismatch");
                }
            }

            return result;
        }
    }

    alsco.types = alsco.types || {};
    alsco.types.Transport = Transport;
    alsco.types.Audio = Audio;
    alsco.types.BoundingBox = BoundingBox;
    alsco.types.Measure = Measure;
    alsco.types.Page = Page;
    alsco.types.MeasureAwareAlignment = MeasureAwareAlignment;
    alsco.types.Project = Project;
})(JSZip, window.alsco);

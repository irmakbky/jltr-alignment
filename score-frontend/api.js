window.alsco = window.alsco || {};

(function (JSZip, alsco) {
    const REMOTE_API_ADDRESS = "http://treble.cs.cmu.edu:5000";

    function getApiAddress() {
        let result = REMOTE_API_ADDRESS;
        if (window.location.href.startsWith("http://treble.cs.cmu.edu")) {
            result = "";
        }
        console.log(`API address: ${result}`);
        return result;
    }

    function fetchJson(addr, body, headers) {
        return fetch(addr, {
            method: "POST",
            headers: headers,
            body: body,
        })
            .then((response) =>
                Promise.all([!response.ok, response.status, response.json()])
            )
            .then((parsed) => {
                const [err, status, result] = parsed;
                if (err) {
                    return Promise.reject(
                        new Error(`HTTP error (${status}): ${result.error}`)
                    );
                }
                return result;
            });
    }

    function fetchFile(url) {
        return fetch(url).then((response) => {
            if (!response.ok) {
                return Promise.reject(
                    new Error(
                        `HTTP error (${response.status}): ${response.error}`
                    )
                );
            }
            return response.blob();
        });
    }

    function callApiPdfFileToPageImages(pdfFile, dpi = 200) {
        const formData = new FormData();
        formData.append("pdf", pdfFile);
        formData.append("dpi", dpi);
        return fetchJson(`${getApiAddress()}/pdf_to_images`, formData).then(
            (o) => o.images.map(alsco.helpers.base64ToBytes)
        );
    }

    function callDetectMeasures(pageImage) {
        const formData = new FormData();
        formData.append("image", new Blob([pageImage], { type: "image/png" }));
        return fetchJson(`${getApiAddress()}/detect_measures`, formData).then(
            (o) => o.bounding_boxes
        );
    }

    function _callUploadProject(project) {
        const formData = new FormData();
        return project
            .toZip()
            .generateAsync({ type: "blob" })
            .then((blob) => {
                formData.append("project", blob);
                return fetchJson(
                    `${getApiAddress()}/upload_project`,
                    formData
                ).then((o) => o.uuid);
            });
    }

    function _callAlign(uuid, hardConstraints) {
        const body = {
            uuid: uuid,
            hard_constraints: hardConstraints,
        };
        return fetchJson(`${getApiAddress()}/align`, JSON.stringify(body), {
            "Content-Type": "application/json",
        }).then((o) => o.alignment);
    }

    async function callAlign(project, uuid, hardConstraints) {
        const numRetries = uuid === null ? 0 : 1;
        for (let i = 0; i <= numRetries; i++) {
            if (uuid === null) {
                uuid = await _callUploadProject(project);
            }
            try {
                return [uuid, await _callAlign(uuid, hardConstraints)];
            } catch (error) {
                if (error.response && error.response.status === 404) {
                    uuid = null;
                } else {
                    throw error;
                }
            }
        }
        throw new Error("Failed to align project.");
    }

    alsco.api = alsco.api || {};
    alsco.api.callApiPdfFileToPageImages = callApiPdfFileToPageImages;
    alsco.api.callDetectMeasures = callDetectMeasures;
    alsco.api.callAlign = callAlign;
    alsco.api.fetchFile = fetchFile;
})(JSZip, window.alsco);

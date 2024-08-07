window.alsco = window.alsco || {};

(function (CryptoJS, alsco) {
    function base64ToBytes(base64) {
        const binary_string = window.atob(base64);
        const len = binary_string.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binary_string.charCodeAt(i);
        }
        return bytes;
    }

    function bytesToBase64(bytes) {
        let binary = "";
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return window.btoa(binary);
    }

    function openFilePromise(ext) {
        return new Promise((resolve, reject) => {
            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.style.display = "none";
            fileInput.accept = ext;
            document.body.appendChild(fileInput);
            fileInput.onchange = function (e) {
                if (this.files.length > 0) {
                    resolve(this.files[0]);
                } else {
                    reject(new Error("No file selected"));
                }
                document.body.removeChild(fileInput);
            };
            fileInput.click();
        });
    }

    function sortObjToHash(attrs) {
        if (attrs === null || attrs === undefined) {
            throw new Error("Leaf cannot be None");
        } else if (typeof attrs === "object") {
            if (Array.isArray(attrs)) {
                return attrs.map(sortObjToHash);
            } else if (attrs instanceof Object) {
                return Object.entries(attrs)
                    .sort(([keyA], [keyB]) => keyA.localeCompare(keyB))
                    .filter(
                        ([, value]) => value !== undefined && value !== null
                    )
                    .map(([key, value]) => [key, sortObjToHash(value)]);
            }
        } else if (typeof attrs === "string" || typeof attrs === "number") {
            return attrs;
        }
        throw new TypeError(`Invalid type: ${typeof attrs} ${attrs}`);
    }

    function sha256(data) {
        if (typeof data === "string") {
            return CryptoJS.SHA256(data).toString(CryptoJS.enc.Hex);
        } else if (data instanceof Uint8Array) {
            const wordArray = CryptoJS.lib.WordArray.create(data);
            return CryptoJS.SHA256(wordArray).toString(CryptoJS.enc.Hex);
        } else {
            throw new Error(
                "Data type not supported. Please provide a string or Uint8Array."
            );
        }
    }

    function hashObj(obj) {
        return sha256(JSON.stringify(sortObjToHash(obj), null, 4));
    }

    function downloadJSON(jsonStr, filename = "data.json") {
        // Create a Blob from the JSON data
        const blob = new Blob([jsonStr], {
            type: "application/json",
        });

        // Create a link element
        const link = document.createElement("a");

        // Create a URL for the Blob and set it as the href attribute
        link.href = URL.createObjectURL(blob);

        // Set the download attribute with the filename
        link.download = filename;

        // Append the link to the body (required for Firefox)
        document.body.appendChild(link);

        // Programmatically click the link to trigger the download
        link.click();

        // Remove the link from the document
        document.body.removeChild(link);
    }

    alsco.helpers = alsco.helpers || {};
    alsco.helpers.base64ToBytes = base64ToBytes;
    alsco.helpers.bytesToBase64 = bytesToBase64;
    alsco.helpers.openFilePromise = openFilePromise;
    alsco.helpers._sortObjToHash = sortObjToHash;
    alsco.helpers.sha256 = sha256;
    alsco.helpers.hashObj = hashObj;
    alsco.helpers.downloadJSON = downloadJSON;
})(window.CryptoJS, window.alsco);

import QtQml

QtObject {
    id: root

    property var exportQueueById: ({})
    property var exportPreviewRows: []
    readonly property int exportQueueCount: Object.keys(exportQueueById).length

    function keyForElement(elementId) {
        return String(Number(elementId))
    }

    function exportStatusKey(elementId, imageId) {
        return String(Number(elementId)) + ":" + String(Number(imageId))
    }

    function addTargets(items) {
        if (!items || items.length === 0) {
            return
        }

        const next = Object.assign({}, exportQueueById)
        for (let i = 0; i < items.length; ++i) {
            const item = items[i]
            next[keyForElement(item.elementId)] = {
                elementId: Number(item.elementId),
                imageId: Number(item.imageId),
                fileName: item.fileName ? item.fileName : qsTr("(unnamed)")
            }
        }
        exportQueueById = next
        refreshExportPreview()
    }

    function clearQueue() {
        exportQueueById = ({})
        refreshExportPreview()
    }

    function pruneDeletedElements(elementIds) {
        if (!elementIds || elementIds.length === 0) {
            return
        }

        const deleted = {}
        for (let i = 0; i < elementIds.length; ++i) {
            deleted[keyForElement(elementIds[i])] = true
        }

        const nextQueue = {}
        const queueRows = Object.values(exportQueueById)
        for (let i = 0; i < queueRows.length; ++i) {
            const row = queueRows[i]
            const key = keyForElement(row.elementId)
            if (!deleted[key]) {
                nextQueue[key] = row
            }
        }
        exportQueueById = nextQueue
        refreshExportPreview()
    }

    function pruneCompleted(statusMap) {
        if (!statusMap) {
            return
        }

        const rows = Object.values(exportQueueById)
        if (rows.length === 0) {
            return
        }

        let removed = 0
        const next = {}
        for (let i = 0; i < rows.length; ++i) {
            const row = rows[i]
            const status = String(statusMap[exportStatusKey(row.elementId, row.imageId)] || "")
            if (status === "succeeded" || status === "failed") {
                removed += 1
                continue
            }
            next[keyForElement(row.elementId)] = row
        }

        if (removed > 0) {
            exportQueueById = next
            refreshExportPreview()
        }
    }

    function exportQueueTargets() {
        const rows = Object.values(exportQueueById)
        const targets = []
        for (let i = 0; i < rows.length; ++i) {
            targets.push({
                elementId: rows[i].elementId,
                imageId: rows[i].imageId
            })
        }
        return targets
    }

    function refreshExportPreview() {
        const src = Object.values(exportQueueById)
        if (src.length <= 200) {
            src.sort((a, b) => String(a.fileName).localeCompare(String(b.fileName)))
        }
        const next = []
        for (let i = 0; i < src.length; ++i) {
            const item = src[i]
            next.push({
                statusKey: exportStatusKey(item.elementId, item.imageId),
                summaryRow: false,
                label: item.fileName ? item.fileName : qsTr("(unnamed)")
            })
        }
        exportPreviewRows = next
    }
}

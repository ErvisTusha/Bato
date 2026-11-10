"""Curated HTTP sources; custom manifests use these same asset fields."""
from urllib.parse import quote

POI_REVISION = "ae62bb5116b9aee19ebd5834e3a82066132c9f7f"


def asset(identifier, category, url, destination, prepared_bytes, transfer_bytes=None, **extra):
    return dict(id=identifier, category=category, urls=[url], destination=destination,
                prepared_bytes=prepared_bytes, transfer_bytes=transfer_bytes or prepared_bytes,
                origin="real", version=extra.pop("version", url), **extra)


def default_catalog():
    assets = [
        asset("silesia", "compression", "https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip", "real/compression/silesia", 230_000_000, 68_182_744, archive="zip", expected_size=68_182_744),
        asset("canterbury", "compression", "https://corpus.canterbury.ac.nz/resources/cantrbry.tar.gz", "real/compression/canterbury", 5_000_000, 739_071, archive="tar", expected_size=739_071),
        asset("calgary", "compression", "https://corpus.canterbury.ac.nz/resources/calgary.tar.gz", "real/compression/calgary", 4_000_000, 1_070_276, archive="tar", expected_size=1_070_276),
        asset("enwik8", "text", "https://mattmahoney.net/dc/enwik8.zip", "real/text/enwik8", 100_000_000, 40_000_000, archive="zip"),
        asset("curl-source", "source", "https://curl.se/download/curl-8.10.1.tar.xz", "real/source/curl-8.10.1", 40_000_000, 5_000_000, archive="tar", version="8.10.1"),
        asset("zlib-git", "git", "https://github.com/madler/zlib.git", "real/git/zlib", 30_000_000, kind="git", revision="v1.3.1", version="v1.3.1", depth=100),
        asset("bbb-audio", "audio", "https://media.xiph.org/BBB/BigBuckBunny-stereo.flac", "real/audio/big_buck_bunny.flac", 220_000_000, validation="flac"),
        asset("akiyo-video", "video", "https://media.xiph.org/video/derf/y4m/akiyo_qcif.y4m", "real/video/akiyo_qcif.y4m", 12_000_000),
        asset("foreman-video", "video", "https://media.xiph.org/video/derf/y4m/foreman_cif.y4m", "real/video/foreman_cif.y4m", 50_000_000),
        asset("alpine-iso", "iso", "https://dl-cdn.alpinelinux.org/alpine/v3.20/releases/x86_64/alpine-standard-3.20.3-x86_64.iso", "real/iso/alpine-standard-3.20.3.iso", 250_000_000, validation="iso", version="3.20.3"),
        asset("enwik9", "large", "https://mattmahoney.net/dc/enwik9.zip", "real/large/enwik9", 1_000_000_000, 350_000_000, archive="zip", tier="large"),
        asset("linux-6.6", "large", "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.6.tar.xz", "real/large/linux-6.6.tar", 1_600_000_000, 150_000_000, archive="xz", tier="large", version="6.6"),
        asset("ubuntu-iso", "large", "https://releases.ubuntu.com/22.04/ubuntu-22.04.5-live-server-amd64.iso", "real/large/ubuntu-22.04.5.iso", 2_500_000_000, validation="iso", tier="large", version="22.04.5"),
        asset("sintel-mp4", "large", "https://archive.org/download/Sintel/sintel-2048-surround.mp4", "real/large/sintel.mp4", 400_000_000, validation="mp4", tier="large"),
    ]
    for index in range(1, 25):
        assets.append(asset(f"kodak-{index:02d}", "images", f"https://r0k.us/graphics/kodak/kodak/kodim{index:02d}.png", f"real/images/kodak/kodim{index:02d}.png", 2_000_000, validation="png", version="Kodak-24"))
    office = {
        "word": ["SampleDoc.docx", "TestDocument.docx", "Headers.docx", "NoHeadFoot.docx", "HeaderFooterUnicode.docx", "VariousPictures.docx", "footnotes.docx", "documentProperties.docx"],
        "excel": ["SampleSS.xlsx", "SimpleNormal.xlsx", "SimpleStrict.xlsx", "SimpleMultiCell.xlsx", "SimpleWithComments.xlsx", "SimpleScatterChart.xlsx", "WithChart.xlsx", "WithThreeCharts.xlsx"],
        "powerpoint": ["SampleShow.pptx", "sample.pptx", "testPPT.pptx", "table_test.pptx", "table_test2.pptx", "smartart-simple.pptx", "picture-transparency.pptx", "ArtisticEffectSample.pptx"],
    }
    folders = {"word": "document", "excel": "spreadsheet", "powerpoint": "slideshow"}
    for category, names in office.items():
        for name in names:
            url = f"https://raw.githubusercontent.com/apache/poi/{POI_REVISION}/test-data/{folders[category]}/{quote(name)}"
            assets.append(asset(f"poi-{category}-{name}", category, url, f"real/documents/{category}/{name}", 2_000_000, validation=name.rsplit(".", 1)[1], version=POI_REVISION, license="Apache-2.0"))
    for name, url in [("7zr.exe", "https://www.7-zip.org/a/7zr.exe"), ("sigcheck.exe", "https://live.sysinternals.com/sigcheck.exe"), ("k9mail.apk", "https://f-droid.org/repo/com.fsck.k9_39006.apk")]:
        category = "apk" if name.endswith("apk") else "exe"
        assets.append(asset(f"binary-{name}", category, url, f"real/binaries/{name}", 30_000_000, validation=category, version="snapshot-at-download"))
    return assets


ALIASES = {
    "docs": {"word", "excel", "powerpoint"}, "documents": {"word", "excel", "powerpoint"},
    "office": {"word", "excel", "powerpoint"}, "ppt": {"powerpoint"}, "repo": {"git"},
    "jpg": {"images"}, "png": {"images"}, "photos": {"images"}, "mp4": {"video"},
    "binaries": {"exe", "apk", "aab"}, "kernel": {"large"}, "wikipedia": {"text", "large"},
}

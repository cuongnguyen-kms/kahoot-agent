from kahoot_agent.file_reader_tool import read_external_file

def test_read_external_file():
    # Google Drive file: use direct download link
    url = "https://drive.google.com/uc?export=download&id=1X78qNfMWkGRcAEHcss6_w0EgfyEhpJ0V"
    content = read_external_file.invoke({"url": url})
    assert isinstance(content, str)
    print("[Test] File content (first 200 chars):\n", content[:200])
    assert len(content) > 0 and not content.startswith("[File Read Error]")

if __name__ == "__main__":
    test_read_external_file()

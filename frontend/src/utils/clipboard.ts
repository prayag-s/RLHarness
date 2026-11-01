/**
 * Copy that also works when the studio is opened from a non-secure origin
 * (plain http on a LAN address), where `navigator.clipboard` is undefined.
 */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    // Permission denied or blocked — fall through to the legacy path.
  }

  try {
    const node = document.createElement('textarea')
    node.value = text
    node.setAttribute('readonly', '')
    node.style.position = 'fixed'
    node.style.top = '-1000px'
    node.style.opacity = '0'
    document.body.appendChild(node)
    node.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(node)
    return ok
  } catch {
    return false
  }
}

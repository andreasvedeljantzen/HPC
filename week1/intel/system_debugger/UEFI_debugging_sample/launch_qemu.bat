: -----------------------------------------------------------------------------
:        Copyright (C) 2016 Intel Corporation. All rights reserved.
:
: This file is owned by Intel Corporation or its suppliers or licensors,
: and is furnished under license.  It may only be used or copied in accordance
: with the terms of that license.  Unless otherwise provided in that license,
: it is provided AS IS, with no warranties of any kind, express or implied.
: Except as expressly permitted by that license, neither Intel Corporation nor
: its suppliers assumes any responsibility or liability for any errors or
: inaccuracies that may appear herein. Except as expressly permitted by that
: license, no part of the Software may be reproduced, stored in a retrieval
: system, transmitted in any form, or distributed by any means without the
: express written consent of Intel Corporation. Title to this material remains
: with Intel Corporation or its suppliers and licensors.
: -----------------------------------------------------------------------------

set EDKII_BUILD_OUTPUT=%~dp0

start /b "" "C:\Program Files\qemu\qemu-system-i386w.exe" ^
-drive if=pflash,format=raw,unit=0,file=%EDKII_BUILD_OUTPUT%FV\OVMF_CODE.fd,readonly=on ^
-drive if=pflash,format=raw,unit=1,file=%EDKII_BUILD_OUTPUT%FV\OVMF_VARS.fd ^
-serial tcp:127.0.0.1:20716,server

exit
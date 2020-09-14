/ (fcn) entry0 533
|   entry0 ();
|           ; var int32_t var_24h @ ebp-0x24
|           ; var int32_t var_20h @ ebp-0x20
|           ; var int32_t var_18h @ ebp-0x18
|           ; var uint32_t var_14h @ ebp-0x14
|           ; var int32_t var_10h @ ebp-0x10
|           0x0040aa98      55             push ebp
|           0x0040aa99      8bec           mov ebp, esp
|           0x0040aa9b      83c4c4         add esp, 0xffffffc4
|           0x0040aa9e      53             push ebx
|           0x0040aa9f      56             push esi
|           0x0040aaa0      57             push edi
|           0x0040aaa1      33c0           xor eax, eax
|           0x0040aaa3      8945f0         mov dword [var_10h], eax
|           0x0040aaa6      8945dc         mov dword [var_24h], eax
|           ; CODE XREF from fcn.0040a128 (+0x916)
|           0x0040aaa9      e82e86ffff     call fcn.004030dc
|           0x0040aaae      e83598ffff     call fcn.004042e8
|           0x0040aab3      e89c9bffff     call fcn.00404654
|           ; CODE XREF from fcn.0040a128 (+0x925)
|           0x0040aab8      e8b79fffff     call fcn.00404a74
|           0x0040aabd      e856bfffff     call fcn.00406a18
|           0x0040aac2      e8ede8ffff     call fcn.004093b4
|           0x0040aac7      e854eaffff     call fcn.00409520
|           0x0040aacc      33c0           xor eax, eax
|           0x0040aace      55             push ebp
|           ; CODE XREF from fcn.0040a128 (+0x936)
|           0x0040aacf      6869b14000     push 0x40b169
|           0x0040aad4      64ff30         push dword fs:[eax]
|           0x0040aad7      648920         mov dword fs:[eax], esp
|           ; CODE XREF from fcn.0040a128 (+0x938)
|           0x0040aada      33d2           xor edx, edx
|           0x0040aadc      55             push ebp
|           0x0040aadd      6832b14000     push 0x40b132
|           0x0040aae2      64ff32         push dword fs:[edx]
|           0x0040aae5      648922         mov dword fs:[edx], esp
|           0x0040aae8      a114d04000     mov eax, dword [0x40d014]   ; [0x40d014:4]=0
|           0x0040aaed      e826f5ffff     call fcn.0040a018
|           0x0040aaf2      e811f1ffff     call fcn.00409c08
|           0x0040aaf7      803d34c24000.  cmp byte [0x40c234], 0      ; [0x40c234:1]=0
|       ,=< 0x0040aafe      740c           je 0x40ab0c
|       |   0x0040ab00      e823f6ffff     call fcn.0040a128
|       |   0x0040ab05      33c0           xor eax, eax
|       |   0x0040ab07      e82493ffff     call fcn.00403e30
|       |   ; CODE XREF from entry0 (0x40aafe)
|       `-> 0x0040ab0c      8d55f0         lea edx, [var_10h]
|           0x0040ab0f      33c0           xor eax, eax
|           0x0040ab11      e866c5ffff     call fcn.0040707c
|           0x0040ab16      8b55f0         mov edx, dword [var_10h]
|           0x0040ab19      b830de4000     mov eax, 0x40de30
|           0x0040ab1e      e8c586ffff     call fcn.004031e8
|           0x0040ab23      6a02           push 2                      ; 2
|           0x0040ab25      6a00           push 0
|           0x0040ab27      6a01           push 1                      ; ecx
|           0x0040ab29      8b0d30de4000   mov ecx, dword [0x40de30]   ; [0x40de30:4]=0
|           0x0040ab2f      b201           mov dl, 1
|           0x0040ab31      b808784000     mov eax, 0x407808
|           0x0040ab36      e821ceffff     call fcn.0040795c
|           0x0040ab3b      a334de4000     mov dword [0x40de34], eax   ; [0x40de34:4]=0
|           0x0040ab40      33d2           xor edx, edx
|           0x0040ab42      55             push ebp
|           0x0040ab43      68eab04000     push 0x40b0ea
|           0x0040ab48      64ff32         push dword fs:[edx]
|           0x0040ab4b      648922         mov dword fs:[edx], esp
|           0x0040ab4e      e881f5ffff     call fcn.0040a0d4
|           0x0040ab53      a33cde4000     mov dword [0x40de3c], eax   ; [0x40de3c:4]=0
|           0x0040ab58      a13cde4000     mov eax, dword [0x40de3c]   ; [0x40de3c:4]=0
|           0x0040ab5d      83780c01       cmp dword [eax + 0xc], 1    ; [0xc:4]=-1 ; 1
|       ,=< 0x0040ab61      7548           jne 0x40abab
|       |   0x0040ab63      a13cde4000     mov eax, dword [0x40de3c]   ; [0x40de3c:4]=0
|       |   0x0040ab68      ba28000000     mov edx, 0x28               ; '(' ; 40
|       |   0x0040ab6d      e822d2ffff     call fcn.00407d94
|       |   0x0040ab72      8b153cde4000   mov edx, dword [0x40de3c]   ; [0x40de3c:4]=0
|       |   0x0040ab78      3b4228         cmp eax, dword [edx + 0x28] ; [0x28:4]=-1 ; '(' ; 40
|      ,==< 0x0040ab7b      752e           jne 0x40abab
